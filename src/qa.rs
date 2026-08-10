//! The live AI QA control socket (`--qa [port]`) — the interactive half of
//! the AI QA Lab. A localhost TCP line protocol that lets a scripted driver
//! (an AI agent, `frqa`, or raw `nc`/PowerShell) drive a LIVE session:
//! teleport, look, strafe at a chosen deflection, scrub time-of-day,
//! synthesize the toggle keys, take screenshots to a named path, and read
//! the session's state — so QA iteration on look/feel bugs no longer
//! round-trips through a human at the keyboard.
//!
//! Provenance: a port of diamondmine's QA control socket design (commit
//! 14683e6 there) with the one Windows change — `TcpListener` on 127.0.0.1
//! in place of the Unix-domain socket (Rust std exposes no AF_UNIX on
//! Windows; everything above the socket type is the same `read_line` /
//! `writeln!` surface). Security model: the localhost bind IS the boundary,
//! like the developer console the verbs mirror — no token, no handshake.
//!
//! Protocol: each request is one raw UTF-8 verb line, newline-terminated —
//! `echo pos | frqa` works, and so does any TCP client. Each request gets
//! exactly ONE JSON line back, in order, on the same connection:
//! `{"ok":true,"lines":[[1,"..."]],"ms":0.4}` — levels 0 echo, 1 info,
//! 2 warn, 3 error; `ok` is false iff any line is error-level. Verb
//! dispatch lives in `main.rs`'s session loop (the drain runs once per
//! frame-loop iteration, BEFORE the menu-hold branch, so a held menu still
//! answers); this module owns only the transport.
//!
//! Threading: one `fr-qa` listener thread (nonblocking accept polled at
//! 100 ms against the quit flag), one `fr-qa-conn` thread per connection.
//! The socket threads never touch game state: each request carries its own
//! bounded(1) reply channel and the connection thread blocks on it, so the
//! session answers out-of-band whenever the verb resolves (same-iteration
//! for sync verbs, later for screenshot/sync pendings). A disconnect is a
//! non-event (the dropped sender turns resolution into a discarded send);
//! a session exit with a request in flight answers "session ended" via the
//! dropped-sender error rather than a silent close.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

pub const DEFAULT_PORT: u16 = 4599;

/// One request from a connection thread: the raw verb line plus its own
/// reply channel (bounded(1) — exactly one reply per request, in order).
pub struct QaReq {
    pub line: String,
    pub reply: std::sync::mpsc::SyncSender<String>,
}

/// The request queue the session loop drains once per iteration. A plain
/// `Mutex<VecDeque>` rather than a channel — the house style (flycam's
/// `Arc<Mutex<FlyState>>`), and the drain wants `try`-semantics anyway.
pub type QaQueue = Arc<Mutex<VecDeque<QaReq>>>;

/// Encode one reply. `ok` is false iff any line is error-level (3).
pub fn reply_json(lines: &[(u8, String)], ms: f64) -> String {
    let ok = lines.iter().all(|(l, _)| *l < 3);
    let arr: Vec<serde_json::Value> =
        lines.iter().map(|(l, t)| serde_json::json!([l, t])).collect();
    serde_json::json!({"ok": ok, "lines": arr, "ms": ms}).to_string()
}

pub fn info_reply(msg: &str) -> String {
    reply_json(&[(1, msg.to_string())], 0.0)
}

pub fn err_reply(msg: &str) -> String {
    reply_json(&[(3, msg.to_string())], 0.0)
}

/// Spawn the listener. `Err` on bind failure — the caller prints and the
/// session continues WITHOUT the socket (an armed-but-broken instrument
/// must be loud, never fatal). The quit flag stops the accept loop within
/// ~100 ms; connection threads die with the process (they block in
/// `read_line` on a client that owns the pacing).
pub fn spawn_listener(
    port: u16,
    queue: QaQueue,
    quit: Arc<AtomicBool>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    listener.set_nonblocking(true)?;
    std::thread::Builder::new().name("fr-qa".into()).spawn(move || {
        while !quit.load(Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    let q = queue.clone();
                    let _ = std::thread::Builder::new()
                        .name("fr-qa-conn".into())
                        .spawn(move || handle_conn(stream, q));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) => {
                    eprintln!("qa: accept failed ({e})");
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    })
}

fn handle_conn(stream: TcpStream, queue: QaQueue) {
    // THE WINDOWS ACCEPT TRAP: an accepted socket INHERITS the listener's
    // nonblocking mode here (unlike Linux, where accept resets it — the
    // diamondmine original never needed this line). Left nonblocking, the
    // first read_line races the client's write: WouldBlock reads as Err,
    // the thread breaks, and the client sees an RST — an INTERMITTENT
    // "write failed" keyed to sub-ms timing (measured live: 1 in ~4
    // back-to-back frqa calls).
    let _ = stream.set_nonblocking(false);
    let Ok(rs) = stream.try_clone() else { return };
    let mut reader = BufReader::new(rs);
    let mut writer = stream;
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break, // EOF / error: disconnect is a non-event
            Ok(_) => {}
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let (tx, rx) = std::sync::mpsc::sync_channel::<String>(1);
        queue.lock().unwrap().push_back(QaReq { line, reply: tx });
        let json = rx.recv().unwrap_or_else(|_| err_reply("session ended"));
        if writeln!(writer, "{json}").is_err() || writer.flush().is_err() {
            break;
        }
    }
}

/// The pure half's pins (run by `--check` beside `cli::self_test`): the
/// ok/error level semantics and that a reply is exactly one parseable JSON
/// line — a scripted driver's whole contract.
pub fn self_test() -> Result<(), String> {
    let ok = reply_json(&[(1, "hello".into()), (2, "careful".into())], 1.5);
    let v: serde_json::Value =
        serde_json::from_str(&ok).map_err(|e| format!("reply not JSON: {e}"))?;
    if v["ok"] != serde_json::Value::Bool(true) || v["lines"][0][1] != "hello" {
        return Err("info/warn reply must be ok:true with lines in order".into());
    }
    let bad = reply_json(&[(1, "partial".into()), (3, "boom".into())], 0.0);
    let v: serde_json::Value = serde_json::from_str(&bad).map_err(|e| e.to_string())?;
    if v["ok"] != serde_json::Value::Bool(false) {
        return Err("any error-level line must make ok:false".into());
    }
    if ok.contains('\n') || bad.contains('\n') {
        return Err("a reply must be one line (the read_line framing contract)".into());
    }
    if !err_reply("x").contains("\"ok\":false") || !info_reply("y").contains("\"ok\":true") {
        return Err("err/info reply helpers inverted".into());
    }
    Ok(())
}

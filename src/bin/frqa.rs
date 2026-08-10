//! frqa — the frustracer QA-socket client (diamondmine's `qactl` twin).
//!
//! `frqa [-p port] <verb ...>` joins its arguments into one line, sends it
//! to the live session's `--qa` control socket (127.0.0.1, default 4599),
//! prints the one-line JSON reply raw, and exits 0 if `"ok":true`, 1 if
//! the verb failed, 2 on usage/connect/IO errors. Deliberately NO read
//! timeout — a pending verb (`sync 300`, `screenshot`) blocks until its
//! reply, and the session side carries the 30 s backstop. Not privileged:
//! any TCP client speaking line-in/JSON-line-out works; this just adds
//! exit codes. Pure std — builds standalone on any platform.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

fn usage() -> ! {
    eprintln!(
        "usage: frqa [-p port] <verb ...>\n  verbs: pos | tp x y z [yaw pitch] | look yaw \
         pitch | tod H | drive x y z ticks | drive stop | key <name> | screenshot <path.png> \
         | sync N | quit   (see `frustracer --qa`)"
    );
    std::process::exit(2);
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut port = 4599u16;
    if args.first().map(String::as_str) == Some("-p") {
        if args.len() < 2 {
            usage();
        }
        port = match args[1].parse() {
            Ok(p) if p != 0 => p,
            _ => usage(),
        };
        args.drain(0..2);
    }
    if args.is_empty() {
        usage();
    }
    let line = args.join(" ");
    let stream = match TcpStream::connect(("127.0.0.1", port)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("frqa: connect 127.0.0.1:{port} failed ({e}) — is the session running with --qa?");
            std::process::exit(2);
        }
    };
    let Ok(rs) = stream.try_clone() else {
        eprintln!("frqa: stream clone failed");
        std::process::exit(2);
    };
    let mut reader = BufReader::new(rs);
    let mut writer = stream;
    if writeln!(writer, "{line}").and_then(|()| writer.flush()).is_err() {
        eprintln!("frqa: write failed");
        std::process::exit(2);
    }
    let mut reply = String::new();
    match reader.read_line(&mut reply) {
        Ok(n) if n > 0 => {}
        _ => {
            eprintln!("frqa: read failed (session gone?)");
            std::process::exit(2);
        }
    }
    print!("{reply}");
    std::process::exit(if reply.contains("\"ok\":true") { 0 } else { 1 });
}

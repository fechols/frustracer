//! The Slint markup for the HUD (compass + clock + the motion-gated keymap
//! panel) and, later, the pause menu. Rendered by the SOFTWARE renderer into
//! `Hud`'s CPU buffer — the root window background is transparent, so the 3D
//! frame shows through everywhere the UI doesn't paint.
//!
//! Design constraints the markup honors:
//! - Dirty-rect friendliness: static elements never change properties, the
//!   compass letters move only when the QUANTIZED heading changes (mod.rs
//!   rounds to whole degrees), the clock only on a minute change, and the
//!   keymap panel fades via ONE animated opacity — while settled (0 or
//!   target) it dirties nothing.
//! - No element rotation (the software renderer's rotation support is
//!   Image-only): the compass letters orbit by explicit sin/cos positioning,
//!   and the FPS graph's "angular" accents are axis-aligned L-brackets. Its
//!   glow is a radial-gradient rect — the software renderer silently ignores
//!   drop-shadow-*, so a shadow property here would render NOTHING.
//! - The FPS graph rewrites only its bar rows + two readout strings, at most
//!   once per 125 ms and only while hud-live (mod.rs gates the writes); all
//!   its chrome (glow, brackets, 60-fps line, captions) is static. The mode
//!   pill churns only on SPACE/F render-mode transitions.
//! - Keymap text mirrors src/flycam.rs's actual bindings (WASD/arrows, drag
//!   look, E/Q up/down, Shift/Ctrl+bumpers slow, ,/. + D-pad time of day) —
//!   update BOTH when a binding changes. The pad line also carries the menu
//!   bindings (src/pad.rs: Start toggle; in-menu D-pad/stick nav, A, B).
//! - The menu's `sel-*` properties are the pad/keyboard navigation cursor
//!   (hud/mod.rs owns it): a cursor move flips exactly two elements'
//!   `sel == i` comparisons, so only those two rows/buttons re-rasterize —
//!   the dirty-rect discipline holds under held-repeat navigation.

slint::slint! {
    // One settings row (built by settings::menu_items + menu_value in Rust).
    // `control` picks the row's interaction: "toggle" (click), "cycle"/"step"
    // (< >), "cyclefwd" (>), "text" (TextInput).
    export struct SettingRow {
        id: string,
        label: string,
        value: string,
        restart: bool,
        cli: bool,
        control: string,
    }

    // One FPS-graph bar: rendered-frame FPS plus the frame-generation
    // ADDITION (presented minus rendered, 0 when no FG inserts) — drawn as
    // a stacked pair on the shared 0..120 scale.
    export struct FpsBar {
        base: float,
        fg: float,
    }

    component MenuButton inherits Rectangle {
        in property <string> label;
        // Pad/keyboard navigation cursor (gold border) — independent of the
        // mouse hover tint, so the two input methods never fight.
        in property <bool> selected;
        callback clicked;
        height: 44px;
        border-radius: 8px;
        background: (ta.has-hover || root.selected) ? #3a444ce0 : #262c38e0;
        border-width: root.selected ? 2px : 1px;
        border-color: root.selected ? #ffd24d : #ffffff30;
        ta := TouchArea {
            clicked => {
                root.clicked();
            }
        }
        Text {
            text: root.label;
            color: #f0f0f0;
            font-size: 17px;
            horizontal-alignment: center;
            vertical-alignment: center;
        }
    }

    component ArrowButton inherits Rectangle {
        in property <string> glyph;
        callback clicked;
        width: 26px;
        height: 26px;
        border-radius: 5px;
        background: ta.has-hover ? #3a444ce0 : #1c222cd0;
        ta := TouchArea {
            clicked => {
                root.clicked();
            }
        }
        Text {
            text: root.glyph;
            color: #e8e8e8;
            font-size: 14px;
            horizontal-alignment: center;
            vertical-alignment: center;
        }
    }

    export component HudUi inherits Window {
        background: transparent;

        // Compass heading in degrees, 0 = north (+Z), clockwise, quantized
        // to whole degrees by the caller.
        in property <float> heading: 0.0;
        in property <string> clock: "12:00";
        // Render mode pill ("CPU" | "GPU" | "DXR") — churns only on the
        // SPACE/F transitions (mod.rs guards it like heading/clock).
        in property <string> mode-label: "CPU";
        // The F1/menu HUD toggle (compass + clock).
        in property <bool> hud-on: true;
        // The keymap/controller panel: on while the camera moves (+linger).
        in property <bool> help-on: false;

        // ── Pause menu state (owned by Rust — src/hud/mod.rs mirrors it).
        in property <bool> menu-open: false;
        // "main" | "settings"
        in property <string> menu-page: "main";
        in property <[string]> groups: [];
        in property <string> menu-group: "Display";
        in property <[SettingRow]> rows: [];
        // "resume" | "settings" | "exit" | "back" | "group:<name>"
        callback menu-action(string);
        // (row id, direction ±1)
        callback row-adjust(string, int);
        // (row id, new text) — Text rows, committed on Enter.
        callback text-edited(string, string);
        // ── Pad/keyboard navigation cursor (Rust owns it — hud/mod.rs's Sel;
        // -1 / false = no selection in that region). Gold-border highlights;
        // only the elements whose comparison flips get dirtied.
        in property <int> sel-main: -1;
        in property <int> sel-tab: -1;
        in property <bool> sel-back: false;
        in property <int> sel-row: -1;
        // The settings TextInput's focus, mirrored out (input.rs gates
        // keyboard nav on it: typing must not move the selection).
        callback edit-focus(bool);

        // Compass + clock fade like the keymap panel: awake on camera/TOD
        // activity (mod.rs's linger), asleep when idle. `hud-on` (F1) stays
        // the hard gate; one animated opacity — settled states dirty nothing.
        in property <bool> hud-live: true;

        // FPS graph rows: bucket-average (rendered FPS, FG-added FPS),
        // oldest first, on a FIXED 0..120 scale (60 = the static mid-strip
        // reference line). mod.rs rewrites the rows of ONE persistent
        // VecModel at most once per 125 ms tick, and only while hud-live —
        // a faded HUD's graph is frozen and dirties nothing.
        in property <[FpsBar]> fps-bars: [];
        in property <string> fps-now: "--";
        in property <string> ms-now: "";

        // ── Loading screen (run_window's pre-session loop drives these; the
        // page shows regardless of `hud-on` — a loading screen is not chrome).
        // While `loading`, the compass/graph/keymap are gated off below so the
        // screen owns the frame; a `set_loading(false)` clears it in one dirty
        // rect. `load-frac < 0` => indeterminate (the marquee sweeps instead).
        in property <bool> loading: false;
        in property <string> load-stage: "";   // "island 5 / 7  san-miguel"
        in property <string> load-phase: "";    // "decoding textures"
        in property <string> load-detail: "";   // scene path / island name
        in property <string> load-count: "";     // "128 / 512"
        in property <float> load-frac: -1.0;     // [0,1], or <0 = marquee
        in property <float> load-marquee: 0.0;   // [0,1] Rust-driven sweep

        compass := Rectangle {
            visible: root.hud-on && !root.loading;
            opacity: root.hud-live ? 1.0 : 0.0;
            animate opacity { duration: 400ms; easing: ease-in-out; }
            x: parent.width - 152px;
            y: 16px;
            width: 136px;
            height: 206px;

            rose := Rectangle {
                x: 8px;
                y: 0px;
                width: 120px;
                height: 120px;
                border-radius: 60px;
                background: #10141cB0;
                border-width: 2px;
                border-color: #ffffff50;

                // Cardinal letters orbit the center: bearing b appears at
                // angle (b - heading) from 12 o'clock.
                n := Text {
                    text: "N";
                    color: #ff6060;
                    font-size: 18px;
                    font-weight: 700;
                    x: 60px - self.width / 2 + 44px * Math.sin((0 - root.heading) * 1deg);
                    y: 60px - self.height / 2 - 44px * Math.cos((0 - root.heading) * 1deg);
                }
                e := Text {
                    text: "E";
                    color: #e8e8e8;
                    font-size: 15px;
                    x: 60px - self.width / 2 + 44px * Math.sin((90 - root.heading) * 1deg);
                    y: 60px - self.height / 2 - 44px * Math.cos((90 - root.heading) * 1deg);
                }
                s := Text {
                    text: "S";
                    color: #e8e8e8;
                    font-size: 15px;
                    x: 60px - self.width / 2 + 44px * Math.sin((180 - root.heading) * 1deg);
                    y: 60px - self.height / 2 - 44px * Math.cos((180 - root.heading) * 1deg);
                }
                w := Text {
                    text: "W";
                    color: #e8e8e8;
                    font-size: 15px;
                    x: 60px - self.width / 2 + 44px * Math.sin((270 - root.heading) * 1deg);
                    y: 60px - self.height / 2 - 44px * Math.cos((270 - root.heading) * 1deg);
                }

                // Fixed view-direction marker at 12 o'clock + center dot.
                needle := Rectangle {
                    x: 60px - 2px;
                    y: 6px;
                    width: 4px;
                    height: 16px;
                    background: #ffd24d;
                    border-radius: 2px;
                }
                dot := Rectangle {
                    x: 60px - 3px;
                    y: 60px - 3px;
                    width: 6px;
                    height: 6px;
                    border-radius: 3px;
                    background: #ffd24d;
                }
            }

            clockbg := Rectangle {
                x: 68px - self.width / 2;
                y: 128px;
                width: 84px;
                height: 34px;
                border-radius: 8px;
                background: #10141cB0;
                border-width: 1px;
                border-color: #ffffff30;
                Text {
                    text: root.clock;
                    color: #ffffff;
                    font-size: 20px;
                    font-weight: 600;
                    horizontal-alignment: center;
                    vertical-alignment: center;
                }
            }

            modebg := Rectangle {
                x: 68px - self.width / 2;
                y: 170px;
                width: 84px;
                height: 26px;
                border-radius: 6px;
                background: #10141cB0;
                border-width: 1px;
                border-color: #58f0ff40;
                Text {
                    text: root.mode-label;
                    color: #7df3ff;
                    font-size: 13px;
                    font-weight: 700;
                    horizontal-alignment: center;
                    vertical-alignment: center;
                }
            }
        }

        // FPS graph: a sci-fi frame-rate sparkline under the compass. ALL
        // chrome is STATIC (glow halo, corner brackets, 60-fps line, header
        // captions) — only the bar rows and the two readout strings change,
        // at most once per 125 ms and only while hud-live.
        fpsgraph := Rectangle {
            visible: root.hud-on && !root.loading;
            opacity: root.hud-live ? 1.0 : 0.0;
            animate opacity { duration: 400ms; easing: ease-in-out; }
            x: parent.width - 236px;
            y: 238px;
            width: 220px;
            height: 74px;

            // Fake glow: drop-shadow-* is a no-op in the software renderer,
            // so the halo is an oversized radial-gradient rect behind the
            // body (fpsgraph itself must NOT clip for this to show).
            glow := Rectangle {
                x: -12px;
                y: -12px;
                width: parent.width + 24px;
                height: parent.height + 24px;
                background: @radial-gradient(circle, #58f0ff26 0%, #58f0ff00 70%);
            }
            body := Rectangle {
                width: 100%;
                height: 100%;
                border-radius: 8px;
                background: @linear-gradient(180deg, #16202cC8 0%, #0a0f18C8 100%);
                border-width: 1px;
                border-color: #58f0ff50;
                clip: true;

                Text {
                    text: "FRAME";
                    color: #58f0ff90;
                    font-size: 9px;
                    x: 10px;
                    y: 7px;
                }
                Text {
                    text: root.ms-now;
                    color: #b8c0cc;
                    font-size: 10px;
                    x: 52px;
                    y: 6px;
                }
                Text {
                    text: root.fps-now;
                    color: #7df3ff;
                    font-size: 15px;
                    font-weight: 700;
                    horizontal-alignment: right;
                    width: 56px;
                    x: 138px;
                    y: 2px;
                }
                Text {
                    text: "FPS";
                    color: #58f0ff90;
                    font-size: 9px;
                    x: 198px;
                    y: 8px;
                }

                // Bars: 40 x 4px on a 5px pitch (strip x 10..210), baseline
                // y 66, 44px full scale. base <= 0 (unfilled boot slots)
                // draws nothing; the STACK (base + fg) clamps at 120 fps —
                // the FG segment rides on the base bar's drawn top (its 2px
                // floor included) and shows the presented frames the render
                // loop never sees. Colors: base bands by RENDERED fps (FG
                // doesn't make tracing cheaper); the FG add-on is violet.
                for b[i] in root.fps-bars : Rectangle {
                    x: 10px + i * 5px;
                    y: 22px;
                    width: 4px;
                    height: 44px;
                    basebar := Rectangle {
                        width: 100%;
                        height: b.base <= 0.0 ? 0px : Math.max(2px, Math.min(b.base / 120.0, 1.0) * 44px);
                        y: parent.height - self.height;
                        background: b.base >= 60.0
                            ? @linear-gradient(180deg, #7df3ff 0%, #1e7fa0F0 100%)
                            : (b.base >= 30.0
                                ? @linear-gradient(180deg, #ffd24d 0%, #a06a1eF0 100%)
                                : @linear-gradient(180deg, #ff6060 0%, #a01e2eF0 100%));
                    }
                    Rectangle {
                        width: 100%;
                        height: b.fg <= 0.0 ? 0px : Math.max(0px,
                            Math.min((b.base + b.fg) / 120.0, 1.0) * 44px - basebar.height);
                        y: basebar.y - self.height;
                        background: @linear-gradient(180deg, #e08dff 0%, #8d3fc0F0 100%);
                    }
                }
                // 60-fps reference: mid-strip of the FIXED 0..120 scale —
                // the fixed scale is what keeps this element static.
                budget := Rectangle {
                    x: 10px;
                    y: 44px;
                    width: 200px;
                    height: 1px;
                    background: #ffd24d60;
                }
                Text {
                    text: "60";
                    color: #ffd24d80;
                    font-size: 8px;
                    x: 199px;
                    y: 35px;
                }
            }
            // Angular corner accents: axis-aligned L-brackets (per-element
            // rotation is a no-op in the software renderer).
            Rectangle { x: -3px; y: -3px; width: 14px; height: 2px; background: #58f0ffC0; }
            Rectangle { x: -3px; y: -3px; width: 2px; height: 14px; background: #58f0ffC0; }
            Rectangle { x: parent.width - 11px; y: -3px; width: 14px; height: 2px; background: #58f0ffC0; }
            Rectangle { x: parent.width + 1px; y: -3px; width: 2px; height: 14px; background: #58f0ffC0; }
            Rectangle { x: -3px; y: parent.height + 1px; width: 14px; height: 2px; background: #58f0ffC0; }
            Rectangle { x: -3px; y: parent.height - 11px; width: 2px; height: 14px; background: #58f0ffC0; }
            Rectangle { x: parent.width - 11px; y: parent.height + 1px; width: 14px; height: 2px; background: #58f0ffC0; }
            Rectangle { x: parent.width + 1px; y: parent.height - 11px; width: 2px; height: 14px; background: #58f0ffC0; }
        }

        // Keymap / controller layout: fades IN while the camera is moving
        // (that is when the pilot wants it), lingers briefly, fades OUT at
        // rest. One animated opacity — settled states dirty nothing.
        help := Rectangle {
            x: (parent.width - self.width) / 2;
            y: parent.height - 96px;
            width: 720px;
            height: 76px;
            border-radius: 10px;
            background: #10141cC0;
            border-width: 1px;
            border-color: #ffffff30;
            opacity: (root.help-on && !root.loading) ? 0.92 : 0.0;
            animate opacity { duration: 350ms; easing: ease-in-out; }

            VerticalLayout {
                padding: 10px;
                spacing: 6px;
                Text {
                    text: "W A S D / arrows  fly      drag mouse  look      E / Q  up / down      Shift / Ctrl  slow      , / .  time of day";
                    color: #f0f0f0;
                    font-size: 13px;
                    horizontal-alignment: center;
                }
                Text {
                    text: "pad:  L stick  fly      R stick  look      triggers  up / down      bumpers  slow      D-pad L / R  time of day      Start  menu";
                    color: #b8c0cc;
                    font-size: 13px;
                    horizontal-alignment: center;
                }
            }
        }

        // ── Pause menu: scrim + centered panel. ESC opens/closes (Rust owns
        // the state machine; the scrim TouchArea swallows scene clicks).
        if root.menu-open : Rectangle {
            width: 100%;
            height: 100%;
            background: #060810A8;
            TouchArea {}

            // Main page: Resume / Settings / Exit.
            if root.menu-page == "main" : Rectangle {
                x: (parent.width - self.width) / 2;
                y: (parent.height - self.height) / 2;
                width: 340px;
                height: 300px;
                border-radius: 14px;
                background: #10141cF0;
                border-width: 1px;
                border-color: #ffffff40;
                VerticalLayout {
                    padding: 24px;
                    spacing: 14px;
                    Text {
                        text: "paused";
                        color: #ffffff;
                        font-size: 24px;
                        font-weight: 700;
                        horizontal-alignment: center;
                    }
                    MenuButton {
                        label: "Resume";
                        selected: root.sel-main == 0;
                        clicked => { root.menu-action("resume"); }
                    }
                    MenuButton {
                        label: "Settings";
                        selected: root.sel-main == 1;
                        clicked => { root.menu-action("settings"); }
                    }
                    MenuButton {
                        label: "Exit";
                        selected: root.sel-main == 2;
                        clicked => { root.menu-action("exit"); }
                    }
                }
            }

            // Settings page: group tabs left, rows right.
            if root.menu-page == "settings" : Rectangle {
                x: (parent.width - self.width) / 2;
                y: (parent.height - self.height) / 2;
                width: 860px;
                height: 620px;
                border-radius: 14px;
                background: #10141cF0;
                border-width: 1px;
                border-color: #ffffff40;
                HorizontalLayout {
                    padding: 18px;
                    spacing: 14px;
                    VerticalLayout {
                        width: 170px;
                        spacing: 8px;
                        Text {
                            text: "settings";
                            color: #ffffff;
                            font-size: 20px;
                            font-weight: 700;
                        }
                        for g[i] in root.groups : Rectangle {
                            height: 36px;
                            border-radius: 7px;
                            background: g == root.menu-group ? #3a444ce0
                                : (gta.has-hover ? #2a323ce0 : #1c222c00);
                            // Nav cursor: gold border, distinct from the
                            // active-group background above.
                            border-width: root.sel-tab == i ? 2px : 0px;
                            border-color: #ffd24d;
                            gta := TouchArea {
                                clicked => { root.menu-action("group:" + g); }
                            }
                            Text {
                                text: g;
                                color: #e8e8e8;
                                font-size: 15px;
                                x: 12px;
                                vertical-alignment: center;
                                height: 100%;
                            }
                        }
                        Rectangle {}
                        MenuButton {
                            label: "Back";
                            selected: root.sel-back;
                            clicked => { root.menu-action("back"); }
                        }
                    }
                    Rectangle {
                        // (leading "0b" would lex as a Rust binary literal
                        // inside the slint! macro — keep hex colors off 0b/0x)
                        background: #0d1018c0;
                        border-radius: 10px;
                        flick := Flickable {
                            viewport-height: rowscol.preferred-height + 20px;
                            // Scroll the nav-selected row into view. Row i
                            // spans y = 10px + i*36px .. +34px (the layout's
                            // padding 10 / height 34 / spacing 2 below —
                            // keep in lockstep). Only fires when the cursor
                            // actually crosses a viewport edge.
                            property <int> sr: root.sel-row;
                            changed sr => {
                                if (self.sr >= 0) {
                                    if (10px + self.sr * 36px + self.viewport-y < 0px) {
                                        self.viewport-y = -(10px + self.sr * 36px);
                                    }
                                    if (10px + self.sr * 36px + 34px + self.viewport-y > self.height) {
                                        self.viewport-y = self.height - (10px + self.sr * 36px + 34px);
                                    }
                                }
                            }
                            rowscol := VerticalLayout {
                                padding: 10px;
                                spacing: 2px;
                                for row[i] in root.rows : Rectangle {
                                    height: 34px;
                                    border-radius: 6px;
                                    background: (rta.has-hover || root.sel-row == i)
                                        ? #1c222c80 : transparent;
                                    border-width: root.sel-row == i ? 1px : 0px;
                                    border-color: #ffd24d80;
                                    rta := TouchArea {
                                        clicked => {
                                            if (row.control == "toggle" || row.control == "cyclefwd") {
                                                root.row-adjust(row.id, 1);
                                            }
                                        }
                                    }
                                    HorizontalLayout {
                                        padding-left: 10px;
                                        padding-right: 10px;
                                        spacing: 8px;
                                        Text {
                                            text: row.label;
                                            color: #d8dde5;
                                            font-size: 14px;
                                            vertical-alignment: center;
                                            width: 46%;
                                            overflow: elide;
                                        }
                                        if row.restart : Text {
                                            text: "restart";
                                            color: #ffb04d;
                                            font-size: 11px;
                                            vertical-alignment: center;
                                        }
                                        // A CLI flag overrode this row's saved
                                        // value; the value column then reads
                                        // "saved -> session" (restart rows).
                                        if row.cli : Text {
                                            text: "cli";
                                            color: #4dd2ff;
                                            font-size: 11px;
                                            vertical-alignment: center;
                                        }
                                        Rectangle {}
                                        if row.control == "cycle" || row.control == "step" : ArrowButton {
                                            glyph: "<";
                                            y: 4px;
                                            clicked => { root.row-adjust(row.id, -1); }
                                        }
                                        if row.control != "text" : Text {
                                            text: row.value;
                                            color: #ffffff;
                                            font-size: 14px;
                                            font-weight: 600;
                                            vertical-alignment: center;
                                            horizontal-alignment: right;
                                            width: 170px;
                                            overflow: elide;
                                        }
                                        if row.control == "cycle" || row.control == "step"
                                            || row.control == "cyclefwd" : ArrowButton {
                                            glyph: ">";
                                            y: 4px;
                                            clicked => { root.row-adjust(row.id, 1); }
                                        }
                                        if row.control == "text" : Rectangle {
                                            width: 300px;
                                            height: 26px;
                                            y: 4px;
                                            border-radius: 5px;
                                            background: #060810E0;
                                            border-width: 1px;
                                            border-color: #ffffff30;
                                            ti := TextInput {
                                                x: 6px;
                                                width: parent.width - 12px;
                                                height: 100%;
                                                text: row.value;
                                                color: #ffffff;
                                                font-size: 13px;
                                                single-line: true;
                                                vertical-alignment: center;
                                                accepted => {
                                                    root.text-edited(row.id, self.text);
                                                }
                                                changed has-focus => {
                                                    root.edit-focus(self.has-focus);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Loading screen: full-window scrim + a centered HUD-styled panel
        // reusing the FPS graph's chrome (gradient body, L-bracket accents,
        // radial glow). Topmost child so it covers everything; shown only
        // while `loading`, so a settled session dirties none of it. The colors
        // stay off any leading "0b"/"0x" token (the binary-literal trap above).
        if root.loading : Rectangle {
            width: 100%;
            height: 100%;
            background: #060810F0;
            TouchArea {}

            panel := Rectangle {
                x: (parent.width - self.width) / 2;
                y: (parent.height - self.height) / 2;
                width: 480px;
                height: 200px;
                border-radius: 12px;
                background: @linear-gradient(180deg, #16202cF0 0%, #0a0f18F0 100%);
                border-width: 1px;
                border-color: #58f0ff60;

                // Soft radial glow (drop-shadow is a software-renderer no-op).
                Rectangle {
                    x: 0px;
                    y: 0px;
                    width: 100%;
                    height: 100%;
                    border-radius: 12px;
                    background: @radial-gradient(circle, #1e7fa030 0%, #0a0f1800 70%);
                }

                Text {
                    text: "F R U S T R A C E R";
                    color: #7df3ff;
                    font-size: 22px;
                    font-weight: 700;
                    x: 28px;
                    y: 22px;
                }
                Text {
                    text: root.load-phase;
                    color: #e8f6ff;
                    font-size: 15px;
                    x: 28px;
                    y: 62px;
                }
                Text {
                    text: root.load-stage;
                    color: #9fb4c4;
                    font-size: 13px;
                    horizontal-alignment: right;
                    width: 240px;
                    x: 212px;
                    y: 64px;
                }
                Text {
                    text: root.load-detail;
                    color: #6f8496;
                    font-size: 12px;
                    x: 28px;
                    y: 90px;
                    width: 424px;
                    overflow: elide;
                }

                // Progress track + fill (determinate) or a sweeping marquee
                // segment (indeterminate: load-frac < 0).
                track := Rectangle {
                    x: 28px;
                    y: 128px;
                    width: 424px;
                    height: 8px;
                    border-radius: 4px;
                    background: #1c222c;
                    clip: true;
                    Rectangle {
                        x: root.load-frac >= 0.0
                            ? 0px
                            : (parent.width - 110px) * root.load-marquee;
                        y: 0px;
                        width: root.load-frac >= 0.0
                            ? parent.width * Math.max(0.0, Math.min(root.load-frac, 1.0))
                            : 110px;
                        height: 100%;
                        border-radius: 4px;
                        background: @linear-gradient(90deg, #7df3ff 0%, #1e7fa0F0 100%);
                    }
                }
                Text {
                    text: root.load-count;
                    color: #9fb4c4;
                    font-size: 12px;
                    horizontal-alignment: right;
                    width: 424px;
                    x: 28px;
                    y: 150px;
                }

                // Angular corner accents — axis-aligned L-brackets.
                Rectangle { x: -3px; y: -3px; width: 16px; height: 2px; background: #58f0ffC0; }
                Rectangle { x: -3px; y: -3px; width: 2px; height: 16px; background: #58f0ffC0; }
                Rectangle { x: parent.width - 13px; y: -3px; width: 16px; height: 2px; background: #58f0ffC0; }
                Rectangle { x: parent.width + 1px; y: -3px; width: 2px; height: 16px; background: #58f0ffC0; }
                Rectangle { x: -3px; y: parent.height + 1px; width: 16px; height: 2px; background: #58f0ffC0; }
                Rectangle { x: -3px; y: parent.height - 13px; width: 2px; height: 16px; background: #58f0ffC0; }
                Rectangle { x: parent.width - 13px; y: parent.height + 1px; width: 16px; height: 2px; background: #58f0ffC0; }
                Rectangle { x: parent.width + 1px; y: parent.height - 13px; width: 2px; height: 16px; background: #58f0ffC0; }
            }
        }
    }
}

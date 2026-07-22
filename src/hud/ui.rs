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
//!   Image-only): the compass letters orbit by explicit sin/cos positioning.
//! - Keymap text mirrors src/flycam.rs's actual bindings (WASD/arrows, drag
//!   look, E/Q up/down, Shift/Ctrl+bumpers slow, ,/. + D-pad time of day) —
//!   update BOTH when a binding changes.

slint::slint! {
    // One settings row (built by settings::menu_items + menu_value in Rust).
    // `control` picks the row's interaction: "toggle" (click), "cycle"/"step"
    // (< >), "cyclefwd" (>), "text" (TextInput).
    export struct SettingRow {
        id: string,
        label: string,
        value: string,
        restart: bool,
        control: string,
    }

    component MenuButton inherits Rectangle {
        in property <string> label;
        callback clicked;
        height: 44px;
        border-radius: 8px;
        background: ta.has-hover ? #3a444ce0 : #262c38e0;
        border-width: 1px;
        border-color: #ffffff30;
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

        // Compass + clock fade like the keymap panel: awake on camera/TOD
        // activity (mod.rs's linger), asleep when idle. `hud-on` (F1) stays
        // the hard gate; one animated opacity — settled states dirty nothing.
        in property <bool> hud-live: true;

        compass := Rectangle {
            visible: root.hud-on;
            opacity: root.hud-live ? 1.0 : 0.0;
            animate opacity { duration: 400ms; easing: ease-in-out; }
            x: parent.width - 152px;
            y: 16px;
            width: 136px;
            height: 170px;

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
        }

        // Keymap / controller layout: fades IN while the camera is moving
        // (that is when the pilot wants it), lingers briefly, fades OUT at
        // rest. One animated opacity — settled states dirty nothing.
        help := Rectangle {
            x: (parent.width - self.width) / 2;
            y: parent.height - 96px;
            width: 660px;
            height: 76px;
            border-radius: 10px;
            background: #10141cC0;
            border-width: 1px;
            border-color: #ffffff30;
            opacity: root.help-on ? 0.92 : 0.0;
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
                    text: "pad:  L stick  fly      R stick  look      triggers  up / down      bumpers  slow      D-pad L / R  time of day";
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
                        clicked => { root.menu-action("resume"); }
                    }
                    MenuButton {
                        label: "Settings";
                        clicked => { root.menu-action("settings"); }
                    }
                    MenuButton {
                        label: "Exit";
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
                        for g in root.groups : Rectangle {
                            height: 36px;
                            border-radius: 7px;
                            background: g == root.menu-group ? #3a444ce0
                                : (gta.has-hover ? #2a323ce0 : #1c222c00);
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
                            clicked => { root.menu-action("back"); }
                        }
                    }
                    Rectangle {
                        // (leading "0b" would lex as a Rust binary literal
                        // inside the slint! macro — keep hex colors off 0b/0x)
                        background: #0d1018c0;
                        border-radius: 10px;
                        Flickable {
                            viewport-height: rowscol.preferred-height + 20px;
                            rowscol := VerticalLayout {
                                padding: 10px;
                                spacing: 2px;
                                for row in root.rows : Rectangle {
                                    height: 34px;
                                    border-radius: 6px;
                                    background: rta.has-hover ? #1c222c80 : transparent;
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
}

mod app;
pub mod bresenham;
pub mod rendering;
pub mod scanline;
pub mod triangle;

use crate::app::DickDrawingApp;
use obj::{load_obj, Obj};
use std::fs::File;
use std::io::BufReader;

fn main() {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_decorations(false)
            .with_fullscreen(true),
        ..Default::default()
    };

    eframe::run_native(
        "eframe template",
        native_options,
        Box::new(|cc| Box::new(DickDrawingApp::new(cc))),
    )
    .unwrap();
}

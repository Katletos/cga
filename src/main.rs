mod app;
pub mod bresenham;
pub mod camera;
pub mod scanline;
pub mod triangle;
pub mod obj_file;

use egui::vec2;

use crate::app::DickDrawingApp;
use std::fs::File;
use std::io::BufReader;

fn main() {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(vec2(1024.0, 576.0))
            .with_decorations(true),
        ..Default::default()
    };

    eframe::run_native(
        "c4",
        native_options,
        Box::new(|cc| Box::new(DickDrawingApp::new(cc))),
    )
    .unwrap();
}

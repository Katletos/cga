mod app;

use std::{env, process};
use std::fs::File;
use std::io::BufReader;
use std::process::{exit, ExitStatus};
use obj::{load_obj, Obj};
use crate::app::TemplateApp;

fn main() {
    // let args: Vec<String> = env::args().collect();

    // if args.len() < 2 {
    //     panic!("No input model");
    // }
    // let obj_file= &args[1];

    // let objects = obj::parse(obj_file).expect("Can't parse object.");
    // let object = objects.objects.first().expect("File is empty.");
    // object.geometry.pop().unwrap().shapes.pop().expect()

    //let file = File::open(obj_file).expect("Can't open file");
    //let input = BufReader::new(file);
    //let obj: Obj = load_obj(input).expect("Can't load obg");

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0]),
        ..Default::default()
    };
    eframe::run_native(
        "eframe template",
        native_options,
        Box::new(|cc| Box::new(TemplateApp::new(cc))),
    ).unwrap();
}

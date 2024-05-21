use crate::triangle::ObjTriangle;
use nalgebra::{Point3, Vector3};

#[derive(Default)]
pub struct ObjModel {
    pub vertices: Vec<Point3<f32>>,
    pub normals: Vec<Vector3<f32>>,
    pub texture_coords: Vec<Point3<f32>>,
    pub triangles: Vec<ObjTriangle>,
}

use nalgebra::{Isometry3, Matrix4, Point3, Vector3};

pub struct CameraLocation {
    pub pitch: f32,
    pub yaw: f32,
    pub world_pos: Point3<f32>,
}

impl CameraLocation {
    pub fn new(pitch: f32, yaw: f32, world_pos: Point3<f32>) -> Self {
        Self {
            pitch,
            yaw,
            world_pos,
        }
    }

    pub fn get_look_direction(&self) -> Vector3<f32> {
         Vector3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
    }

    pub fn forward(&mut self){
        self.world_pos += self.get_look_direction();
    }
}

pub struct Camera {
    pub eye: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub look_at: Isometry3<f32>,
}

impl Camera {
    pub fn new(camera_location: &CameraLocation) -> Self {
        let eye = camera_location.world_pos;

        let target = camera_location.world_pos + camera_location.get_look_direction();
        //let target = Point3::new(0.0, 0.0, 0.0);

        let up = Vector3::new(0.0, 1.0, 0.0);

        Camera {
            eye,
            target,
            up,
            look_at: Isometry3::look_at_rh(&eye, &target, &up),
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        let eye = Point3::new(100.0, 100.0, 100.0);
        let target = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::new(0.0, 0.0, 1.0);

        Camera {
            eye,
            target,
            up,
            look_at: Isometry3::look_at_rh(&eye, &target, &up),
        }
    }
}

use app::App;
use tracing::{error, info};

mod app;

pub const RESOURCES_DIR: &str = "res";

fn tracing_init() {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::builder()
            .with_default_directive(tracing::level_filters::LevelFilter::DEBUG.into())
            .from_env_lossy())
        .init();
}

fn peaceful_crash_out(error: &anyhow::Error) {
    error!("Fatal application error: {:#}", error);
    std::process::exit(1);
}

fn safe_main() -> anyhow::Result<()> {
    use winit::dpi::LogicalSize;
    use winit::event::{Event, WindowEvent};
    use winit::event_loop::EventLoop;
    use winit::window::{WindowBuilder};

    info!("Launching voxelement client...");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let mut app = App::create(&window)?;

    info!("Liftoff!");

    event_loop.run(move |event, elwt| {
        match event {
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::RedrawRequested => {
                    app.render(&window).unwrap_or_else(|err| peaceful_crash_out(&err));
                }
                WindowEvent::CloseRequested => {
                    app.destroy();
                    elwt.exit();
                }
                _ => {}
            },
            _ => {}
        }
    })?;

    Ok(())
}

fn main() {
    tracing_init();

    if let Err(err) = safe_main() {
        peaceful_crash_out(&err);
    }
}

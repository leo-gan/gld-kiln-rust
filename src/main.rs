use std::env;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut schema: Option<String> = None;
    let mut out: Option<String> = None;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--schema" => {
                index += 1;
                if index >= args.len() {
                    eprintln!("--schema needs a path");
                    return ExitCode::from(2);
                }
                schema = Some(args[index].clone());
            }
            "--out" => {
                index += 1;
                if index >= args.len() {
                    eprintln!("--out needs a directory");
                    return ExitCode::from(2);
                }
                out = Some(args[index].clone());
            }
            "-h" | "--help" => {
                println!("gld-kilngen-rust --schema FILE --out DIR");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
        index += 1;
    }
    let (Some(schema), Some(out)) = (schema, out) else {
        eprintln!("usage: gld-kilngen-rust --schema FILE --out DIR");
        return ExitCode::from(2);
    };
    match kiln::emit::generate_to_dir(Path::new(&schema), Path::new(&out)) {
        Ok(path) => {
            println!("{}", path.display());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

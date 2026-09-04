fn main() -> anyhow::Result<()> {
    let output = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("visual-qa"));
    pyzhscriber::render_visual_qa(&output)
}

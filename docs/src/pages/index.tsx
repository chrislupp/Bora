import type { ReactNode } from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import styles from "./index.module.css";

/* ── Feature card ── */
type FeatureItem = {
    label: string;
    title: string;
    description: string;
};

const features: FeatureItem[] = [
    {
        label: "SURROGATE",
        title: "NeuralFoil in Rust",
        description:
            "A faithful port of NeuralFoil, the physics-informed neural network surrogate of XFoil. Predicts CL, CD, CM, transition locations, and boundary layer distributions orders of magnitude faster than XFoil, validated against upstream reference data.",
    },
    {
        label: "GRADIENTS",
        title: "Exact Analytic Derivatives",
        description:
            "Forward-mode differentiation through the network yields the gradient of every output with respect to all 23 inputs in a single call. No finite differences, no solver noise, no convergence failures.",
    },
    {
        label: "CONFIDENCE",
        title: "Knows When to Doubt Itself",
        description:
            "Every analysis reports a confidence in [0, 1] based on the distance from the training data. Penalize it in optimization to keep designs inside the region where the surrogate is trustworthy.",
    },
    {
        label: "INTEGRATION",
        title: "Pure Rust, MDO Ready",
        description:
            "HDF5 weights are read with a pure-Rust reader, so there are no C dependencies. The optional philote feature exposes Bora as a Philote discipline served over gRPC.",
    },
];

function FeatureCard({ label, title, description }: FeatureItem) {
    return (
        <div className={styles.featureCard}>
            <span className={styles.featureLabel}>{label}</span>
            <h3 className={styles.featureTitle}>{title}</h3>
            <p className={styles.featureDesc}>{description}</p>
        </div>
    );
}

/* ── Code preview ── */
function CodePreview() {
    const code = `use bora::{AeroInput, ModelSize, NeuralFoil, input_index};

fn main() -> bora::Result<()> {
    let mut input = AeroInput::default();
    input.kulfan.upper_weights =
        [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    input.kulfan.lower_weights =
        [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    input.alpha = 3.0;

    let model = NeuralFoil::new(ModelSize::XLarge)?;
    let result = model.analyze_with_gradients(&input);

    let cl = result.output.cl;
    let dcl_dalpha = result.gradient.cl[input_index::ALPHA];
    println!("CL = {cl:.4}, dCL/dalpha = {dcl_dalpha:.4}");
    Ok(())
}`;

    return (
        <div className={styles.codeBlock}>
            <div className={styles.codeHeader}>
                <span className={styles.codeDotOrange} />
                <span className={styles.codeDotCyan} />
                <span className={styles.codeFilename}>main.rs</span>
            </div>
            <pre className={styles.codePre}>{code}</pre>
        </div>
    );
}

/* ── Stat ── */
function Stat({ value, label }: { value: string; label: string }) {
    return (
        <div className={styles.stat}>
            <div className={styles.statValue}>{value}</div>
            <div className={styles.statLabel}>{label}</div>
        </div>
    );
}

/* ── Hero ── */
function Hero() {
    const { siteConfig } = useDocusaurusContext();
    return (
        <header className={styles.hero}>
            <div className={styles.heroGlow} />
            <div className={styles.heroInner}>
                <div className={styles.heroSuper}>
                    Airfoil Aerodynamics
                </div>
                <h1 className={styles.heroTitle}>{siteConfig.title}</h1>
                <p className={styles.heroTagline}>
                    {siteConfig.tagline}
                </p>
                <div className={styles.heroCtas}>
                    <Link
                        to="/docs/getting-started/installation"
                        className={styles.ctaPrimary}
                    >
                        Get Started
                    </Link>
                    <Link to="/docs/api" className={styles.ctaSecondary}>
                        API Reference
                    </Link>
                </div>
            </div>
        </header>
    );
}

/* ── Page ── */
export default function Home(): ReactNode {
    return (
        <Layout description="Bora - a Rust port of NeuralFoil for fast airfoil analysis with exact analytic gradients.">
            <Hero />

            <section className={styles.statsBar}>
                <div className={styles.statsInner}>
                    <Stat value="Rust" label="Edition 2024" />
                    <Stat value="Cargo" label="Build & Test" />
                    <Stat value="23" label="Differentiable Inputs" />
                    <Stat value="0" label="C Dependencies" />
                </div>
            </section>

            <main className={styles.featuresSection}>
                <div className={styles.featuresInner}>
                    <div className={styles.sectionHeader}>
                        <span className={styles.sectionLabel}>
                            Capabilities
                        </span>
                        <h2 className={styles.sectionTitle}>
                            Built for Airfoil Design Optimization
                        </h2>
                    </div>

                    <div className={styles.featuresGrid}>
                        {features.map((f, i) => (
                            <FeatureCard key={i} {...f} />
                        ))}
                    </div>

                    <div className={styles.codeSection}>
                        <span className={styles.sectionLabelSmall}>
                            Minimal API
                        </span>
                        <CodePreview />
                    </div>
                </div>
            </main>
        </Layout>
    );
}

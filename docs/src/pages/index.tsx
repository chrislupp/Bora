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

// TODO: Replace with your project's actual features after running setup.sh
const features: FeatureItem[] = [
    {
        label: "FEATURE ONE",
        title: "Feature Title",
        description:
            "Description of the first key feature of your library. Explain what it does and why it matters.",
    },
    {
        label: "FEATURE TWO",
        title: "Feature Title",
        description:
            "Description of the second key feature. Highlight the technical advantage or unique approach.",
    },
    {
        label: "FEATURE THREE",
        title: "Feature Title",
        description:
            "Description of the third key feature. Focus on the problem it solves for users.",
    },
    {
        label: "FEATURE FOUR",
        title: "Feature Title",
        description:
            "Description of the fourth key feature. Mention performance, correctness, or ease of use.",
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
    // TODO: Replace with a minimal usage example of your library
    const code = `#include <bora.h>

int main() {
    // Your library usage example here
    return 0;
}`;

    return (
        <div className={styles.codeBlock}>
            <div className={styles.codeHeader}>
                <span className={styles.codeDotOrange} />
                <span className={styles.codeDotCyan} />
                <span className={styles.codeFilename}>main.cpp</span>
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
                {/* TODO: Replace with your project's domain */}
                <div className={styles.heroSuper}>
                    C++20 Library
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
        <Layout description="Bora - A C++20 library. Replace this description after running setup.sh.">
            <Hero />

            {/* TODO: Replace stats with your project's key metrics */}
            <section className={styles.statsBar}>
                <div className={styles.statsInner}>
                    <Stat value="C++20" label="Standard" />
                    <Stat value="CMake" label="Build System" />
                    <Stat value="GTest" label="Test Framework" />
                    <Stat value="MIT" label="License" />
                </div>
            </section>

            <main className={styles.featuresSection}>
                <div className={styles.featuresInner}>
                    <div className={styles.sectionHeader}>
                        {/* TODO: Replace with your project's capabilities heading */}
                        <span className={styles.sectionLabel}>
                            Capabilities
                        </span>
                        <h2 className={styles.sectionTitle}>
                            Built for Your Use Case
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

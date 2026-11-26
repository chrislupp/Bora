import type { ReactNode } from "react";
import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import CodeBlock from "@theme/CodeBlock";
import PolarFigure from "@site/src/components/PolarFigure";
import { useDocPath } from "@site/src/lib/useDocPath";
import styles from "./index.module.css";

const QUICKSTART = `use bora::{AeroInput, ModelSize, NeuralFoil, input_index};

fn main() -> bora::Result<()> {
    let mut input = AeroInput::default();
    input.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    input.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    input.kulfan.leading_edge_weight = 0.02;
    input.kulfan.te_thickness = 0.0025;
    input.alpha = 3.0; // degrees
    input.re = 1e6;

    let model = NeuralFoil::new(ModelSize::XLarge)?;
    let result = model.analyze_with_gradients(&input);

    println!(
        "CL = {:.4}, CD = {:.5}, CM = {:.4}",
        result.output.cl, result.output.cd, result.output.cm
    );
    println!(
        "dCL/dalpha = {:.4} per degree",
        result.gradient.cl[input_index::ALPHA]
    );
    Ok(())
}`;

const OUTPUT = `CL = 0.9516, CD = 0.00729, CM = -0.1301
dCL/dalpha = 0.1019 per degree`;

// The docs, in reading order.
const CONTENTS: [id: string, label: string][] = [
    ["intro", "Overview"],
    ["getting-started/installation", "Installation"],
    ["getting-started/quickstart", "Quickstart"],
    ["guide/geometry", "Airfoil geometry"],
    ["guide/analysis", "Inputs and outputs"],
    ["guide/gradients", "Gradients"],
    ["guide/confidence", "Analysis confidence"],
    ["guide/model-sizes", "Network sizes"],
    ["background/how-it-works", "How NeuralFoil works"],
    ["background/accuracy-and-limits", "Accuracy and limits"],
    ["examples/polar", "Polar sweep"],
    ["examples/philote", "Analysis over Philote"],
    ["examples/openmdao", "Multipoint optimization"],
    ["api/index", "API overview"],
    ["references", "References"],
];

function DocLink({ id, children }: { id: string; children: ReactNode }): ReactNode {
    return <Link to={useDocPath(id)}>{children}</Link>;
}

export default function Home(): ReactNode {
    return (
        <Layout description="Bora is a Rust port of NeuralFoil: airfoil lift, drag, moment and transition in tens of microseconds, with exact analytic gradients.">
            <main className={`bora-book ${styles.page}`}>
                <PolarFigure variant="sheet">
                    <h1>Airfoil characteristics in tens of microseconds, with exact gradients</h1>
                    <p>
                        Bora is a Rust port of Peter Sharpe’s{" "}
                        <Link href="https://github.com/peterdsharpe/NeuralFoil">NeuralFoil</Link>, a
                        neural-network surrogate of XFoil. One call returns lift, drag, moment and
                        transition for an airfoil, with the derivative of every output with respect
                        to all 23 inputs.
                    </p>
                    <p>
                        <DocLink id="getting-started/quickstart">Start with the quickstart</DocLink>
                        , or read <DocLink id="intro">what Bora is and is not</DocLink>.
                    </p>
                </PolarFigure>

                <div className={styles.below}>
                    <p className={styles.margin}>
                        <strong>Figure 1.</strong> The quickstart section at Reynolds numbers of 0.2
                        (circles), 1.0 (triangles) and 5.0 million (squares), computed by Bora with
                        NeuralFoil’s xlarge network. The mark is the quickstart condition, 3° at one
                        million; the heavy line through it is the analytic slope returned with that
                        evaluation.
                    </p>

                    <article className={`markdown ${styles.column}`}>
                        <h2>One call</h2>
                        <p>
                            Describe the section with 18 Kulfan parameters, set the angle of attack
                            and Reynolds number, and analyze. Nothing iterates, so the call cannot
                            fail to converge and costs the same every time.
                        </p>
                        <CodeBlock language="rust" title="src/main.rs">
                            {QUICKSTART}
                        </CodeBlock>
                        <CodeBlock language="text">{OUTPUT}</CodeBlock>
                        <p>These are the values at the mark in Figure 1.</p>

                        <h2>What it covers</h2>
                        <ul>
                            <li>Lift, drag and quarter-chord moment coefficients.</li>
                            <li>The transition location on each surface.</li>
                            <li>
                                Momentum thickness, shape factor and edge velocity at 32 stations
                                per surface.
                            </li>
                            <li>
                                A confidence that falls as the query leaves the conditions the model
                                was trained on.
                            </li>
                            <li>
                                Exact derivatives of all of the above, by forward or reverse mode.
                            </li>
                            <li>
                                Eight network sizes, from 5 to 356 microseconds per analysis on an
                                Apple M1 Pro.
                            </li>
                        </ul>

                        <h2>What it leaves out</h2>
                        <ul>
                            <li>
                                Compressibility. The model is incompressible and has no Mach number
                                input.
                            </li>
                            <li>
                                Deep stall. Results are meaningful within roughly 20° of zero angle
                                of attack.
                            </li>
                            <li>
                                Anything beyond XFoil. The network learned from XFoil, so XFoil’s
                                assumptions are its ceiling.
                            </li>
                        </ul>
                        <p>
                            <DocLink id="background/accuracy-and-limits">
                                Accuracy and limits
                            </DocLink>{" "}
                            gives the measured errors and the conditions to avoid.
                        </p>

                        <h2>Whose model this is</h2>
                        <p>
                            The model, its training and its weights are the work of Peter Sharpe,
                            described in P. Sharpe and R. J. Hansman, “NeuralFoil: An Airfoil
                            Aerodynamics Analysis Tool Using Physics-Informed Machine Learning,”{" "}
                            <Link href="https://arxiv.org/abs/2503.16323">arXiv:2503.16323</Link>,
                            2025. Bora reimplements the evaluation in Rust, adds hand-derived
                            derivatives, and is tested against Python NeuralFoil 0.3.3 to a relative
                            tolerance of 10⁻⁶.
                        </p>
                        <p>
                            Bora was written for{" "}
                            <Link href="https://github.com/chrislupp/Zephyr">Zephyr</Link>, a
                            vortex-lattice code that needs sectional data with derivatives inside
                            its viscous coupling. For most users,{" "}
                            <Link href="https://github.com/peterdsharpe/NeuralFoil">
                                NeuralFoil
                            </Link>{" "}
                            itself will be the better fit.
                        </p>
                        <p>
                            If Bora contributes to published work, cite the NeuralFoil paper.{" "}
                            <DocLink id="references">References</DocLink> has the entries.
                        </p>
                    </article>

                    <nav className={styles.contents} aria-label="Documentation contents">
                        <ol>
                            {CONTENTS.map(([id, label]) => (
                                <li key={id}>
                                    <DocLink id={id}>{label}</DocLink>
                                </li>
                            ))}
                        </ol>
                    </nav>
                </div>
            </main>
        </Layout>
    );
}

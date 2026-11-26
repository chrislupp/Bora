import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

// Every chapter stays open, so the rail always shows the whole contents.
const chapter = (label: string, items: string[]) => ({
    type: "category" as const,
    label,
    collapsible: false,
    items,
});

const sidebars: SidebarsConfig = {
    docsSidebar: [
        chapter("Getting started", [
            "intro",
            "getting-started/installation",
            "getting-started/quickstart",
        ]),
        chapter("Guide", [
            "guide/geometry",
            "guide/analysis",
            "guide/corrections",
            "guide/gradients",
            "guide/confidence",
            "guide/model-sizes",
            "guide/deployment",
            "guide/philote",
        ]),
        chapter("Background", ["background/how-it-works", "background/accuracy-and-limits"]),
        chapter("Examples", ["examples/analysis",
            "examples/polar",
            "examples/philote",
            "examples/openmdao",
        ]),
        "references",
    ],
    apiSidebar: ["api/index"],
};

export default sidebars;

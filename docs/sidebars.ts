import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
    docsSidebar: [
        {
            type: "category",
            label: "Getting Started",
            items: [
                "getting-started/installation",
                "getting-started/quickstart",
            ],
        },
        {
            type: "category",
            label: "Guide",
            items: [
                "guide/concepts",
            ],
        },
        {
            type: "category",
            label: "Examples",
            items: [
                "examples/basic-usage",
            ],
        },
    ],
    apiSidebar: [
        "api/index",
    ],
};

export default sidebars;

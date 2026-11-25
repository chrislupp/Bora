import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { boraLight, boraDark } from "./src/prism";

const config: Config = {
    title: "Bora",
    tagline: "Airfoil aerodynamics from NeuralFoil, in Rust, with exact gradients",
    favicon: "img/favicon.svg",

    future: {
        v4: true,
    },

    url: "https://chrislupp.github.io",
    baseUrl: "/Bora/",

    organizationName: "chrislupp",
    projectName: "Bora",

    onBrokenLinks: "throw",
    onBrokenAnchors: "throw",

    markdown: {
        hooks: {
            onBrokenMarkdownLinks: "throw",
        },
    },

    i18n: {
        defaultLocale: "en",
        locales: ["en"],
    },

    plugins: [
        // `import source from "path?raw"` yields a file's text, so pages can list
        // the repository's example scripts in full without copying them.
        () => ({
            name: "raw-source",
            configureWebpack: () => ({
                module: { rules: [{ resourceQuery: /raw/, type: "asset/source" }] },
            }),
        }),
    ],

    presets: [
        [
            "classic",
            {
                docs: {
                    sidebarPath: "./sidebars.ts",
                    remarkPlugins: [remarkMath],
                    rehypePlugins: [rehypeKatex],
                    editUrl:
                        "https://github.com/chrislupp/Bora/tree/develop/docs/",
                    // The banner already says these docs are unreleased.
                    versions: {
                        current: { badge: false },
                    },
                },
                blog: false,
                theme: {
                    // KaTeX's stylesheet and fonts are bundled from the installed
                    // package, so they always match the version that renders the math.
                    customCss: ["./node_modules/katex/dist/katex.min.css", "./src/css/custom.css"],
                },
            } satisfies Preset.Options,
        ],
    ],

    themeConfig: {
        colorMode: {
            defaultMode: "light",
            respectPrefersColorScheme: true,
        },
        navbar: {
            title: "Bora",
            items: [
                {
                    type: "docSidebar",
                    sidebarId: "docsSidebar",
                    position: "left",
                    label: "Docs",
                },
                {
                    type: "docSidebar",
                    sidebarId: "apiSidebar",
                    position: "left",
                    label: "API",
                },
                {
                    type: "docsVersionDropdown",
                    position: "right",
                },
                {
                    href: "https://github.com/chrislupp/Bora",
                    label: "GitHub",
                    position: "right",
                },
            ],
        },
        footer: {
            copyright: `Copyright ${new Date().getFullYear()} Christopher A. Lupp. Third-party attributions, including NeuralFoil and its network weights, are listed in the <a href="https://github.com/chrislupp/Bora/blob/main/NOTICE">NOTICE</a> file.`,
        },
        prism: {
            theme: boraLight,
            darkTheme: boraDark,
            additionalLanguages: ["rust", "toml", "bash"],
        },
    } satisfies Preset.ThemeConfig,
};

export default config;

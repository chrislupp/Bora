import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const config: Config = {
    title: "Bora",
    tagline:
        "A C++20 library. Replace this tagline after running setup.sh.",
    favicon: "img/favicon.ico",

    future: {
        v4: true,
    },

    // TODO: Update these after running setup.sh
    url: "https://your-github-username.github.io",
    baseUrl: "/Bora/",

    organizationName: "your-github-username",
    projectName: "Bora",

    onBrokenLinks: "throw",

    i18n: {
        defaultLocale: "en",
        locales: ["en"],
    },

    stylesheets: [
        {
            href: "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
            type: "text/css",
            integrity:
                "sha384-nB0miv6/jRmo5RLHO8BIp/8hwC1slNFDuv3VUgI4A7O/lEKmpRgiuNOQI2bRpFB",
            crossorigin: "anonymous",
        },
    ],

    presets: [
        [
            "classic",
            {
                docs: {
                    sidebarPath: "./sidebars.ts",
                    remarkPlugins: [remarkMath],
                    rehypePlugins: [rehypeKatex],
                    // TODO: Update repository URL after running setup.sh
                    editUrl:
                        "https://github.com/your-github-username/Bora/tree/develop/docs/",
                },
                blog: false,
                theme: {
                    customCss: "./src/css/custom.css",
                },
            } satisfies Preset.Options,
        ],
    ],

    themeConfig: {
        colorMode: {
            defaultMode: "dark",
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
                    // TODO: Update repository URL after running setup.sh
                    href: "https://github.com/your-github-username/Bora",
                    label: "GitHub",
                    position: "right",
                },
            ],
        },
        footer: {
            style: "dark",
            links: [
                {
                    title: "Documentation",
                    items: [
                        {
                            label: "Getting Started",
                            to: "/docs/getting-started/installation",
                        },
                        {
                            label: "API Reference",
                            to: "/docs/api",
                        },
                    ],
                },
                {
                    title: "More",
                    items: [
                        {
                            // TODO: Update repository URL after running setup.sh
                            label: "GitHub",
                            href: "https://github.com/your-github-username/Bora",
                        },
                        {
                            label: "Changelog",
                            href: "https://github.com/your-github-username/Bora/blob/main/CHANGELOG.md",
                        },
                    ],
                },
            ],
            copyright: `Copyright \u00A9 ${new Date().getFullYear()} Your Name. Built with Docusaurus.`,
        },
        prism: {
            theme: prismThemes.github,
            darkTheme: prismThemes.dracula,
            additionalLanguages: ["cmake", "bash"],
        },
    } satisfies Preset.ThemeConfig,
};

export default config;

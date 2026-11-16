import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from '@playwright/test';

// The tests exercise a built site: `web/dist` by default, or the directory
// named by YOKAI_WEB_ROOT, for example a release archive extracted elsewhere.
const here = path.dirname(fileURLToPath(import.meta.url));
const siteRoot = path.resolve(process.env.YOKAI_WEB_ROOT ?? path.join(here, '..', 'dist'));
if (!existsSync(path.join(siteRoot, 'index.html'))) {
    throw new Error(`No built site in ${siteRoot}: run ./web/scripts/build.sh first.`);
}

const port = 8473;

// Both projects use the Chromium that Playwright downloads, in its headless
// mode: no window opens. `channel: 'chromium'` selects the complete browser
// instead of the reduced "headless shell", which has no access to the
// graphics card.
const chromium = { browserName: 'chromium', channel: 'chromium' };

export default defineConfig({
    testDir: here,
    outputDir: path.join(here, 'test-results'),
    fullyParallel: true,
    forbidOnly: Boolean(process.env.CI),
    // A two-player game takes more than a minute at the pace of its
    // animations.
    timeout: 5 * 60 * 1000,
    expect: { timeout: 15 * 1000 },
    // The HTML reporter would open the report in the default browser.
    reporter: 'list',
    use: {
        baseURL: `http://127.0.0.1:${port}/`,
        trace: 'retain-on-failure',
        screenshot: 'only-on-failure'
    },
    // `backend` is the engine module that the page is expected to load in a
    // project; the tests read it and check the `pkg-…` directory requested.
    projects: [
        {
            // A desktop window driven with a mouse, on the WebGPU module.
            name: 'desktop-webgpu',
            metadata: { backend: 'webgpu' },
            use: { ...chromium, viewport: { width: 1280, height: 800 } }
        },
        {
            // A phone held upright, driven by touch. Without a graphics
            // adapter the page falls back to the Flex module, which runs on
            // the processor.
            name: 'mobile-flex',
            metadata: { backend: 'flex' },
            use: {
                ...chromium,
                viewport: { width: 390, height: 844 },
                deviceScaleFactor: 3,
                isMobile: true,
                hasTouch: true,
                launchOptions: { args: ['--disable-gpu'] }
            }
        }
    ],
    webServer: {
        command: `node "${path.join(here, 'serve.js')}" "${siteRoot}" ${port}`,
        url: `http://127.0.0.1:${port}/`,
        // A server left by another run could serve another directory.
        reuseExistingServer: false
    }
});

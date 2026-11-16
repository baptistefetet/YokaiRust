// Build suffix of the page (see main.js), received through the worker's own
// address. The model and the engine module must come from the same build.
const BUILD_QUERY = self.location.search;

let rustGame = null;
let backend = null;

self.addEventListener('message', async (event) => {
    const { id, command, payload } = event.data;
    try {
        let result;
        if (command === 'init') {
            result = await initialize(payload);
        } else if (command === 'snapshot') {
            requireGame();
            result = parseJson(rustGame.snapshotJson());
        } else if (command === 'reset') {
            requireGame();
            result = parseJson(rustGame.reset(payload.humanPlayers));
        } else if (command === 'human') {
            requireGame();
            result = parseJson(rustGame.playHuman(JSON.stringify(payload.action)));
        } else if (command === 'ai') {
            requireGame();
            result = parseJson(await rustGame.playAi());
        } else {
            throw new Error(`Unknown engine command: ${command}`);
        }
        self.postMessage({ id, result });
    } catch (error) {
        self.postMessage({ id, error: errorMessage(error) });
    }
});

async function initialize({ simulations, humanPlayers }) {
    const [modelResponse, metadataResponse] = await Promise.all([
        fetch(`./model/champion.bin${BUILD_QUERY}`),
        fetch(`./model/metadata.json${BUILD_QUERY}`)
    ]);
    if (!modelResponse.ok) {
        throw new Error(`Model download failed (${modelResponse.status})`);
    }
    if (!metadataResponse.ok) {
        throw new Error(`Metadata download failed (${metadataResponse.status})`);
    }

    const modelBytes = new Uint8Array(await modelResponse.arrayBuffer());
    const metadataJson = await metadataResponse.text();
    const attempts = await webGpuAdapterAvailable() ? ['webgpu', 'flex'] : ['flex'];
    const failures = [];

    for (const candidate of attempts) {
        try {
            const directory = `./pkg-${candidate}`;
            const module = await import(`${directory}/yokai_web.js${BUILD_QUERY}`);
            // Without an explicit address the bindings would request the
            // WebAssembly file without the build suffix.
            await module.default({
                module_or_path: new URL(`${directory}/yokai_web_bg.wasm${BUILD_QUERY}`, import.meta.url)
            });
            rustGame = await module.createGame(
                modelBytes,
                metadataJson,
                simulations,
                humanPlayers
            );
            backend = module.backendName();
            return {
                backend,
                state: parseJson(rustGame.snapshotJson())
            };
        } catch (error) {
            failures.push(`${candidate}: ${errorMessage(error)}`);
        }
    }

    throw new Error(`No browser inference backend could start. ${failures.join(' / ')}`);
}

// A browser can expose `navigator.gpu` and still have no usable adapter
// (headless sessions, blocked drivers). The WebGPU module panics in that case
// instead of failing a promise, so the adapter is requested here first.
async function webGpuAdapterAvailable() {
    if (!('gpu' in self.navigator)) {
        return false;
    }
    try {
        return await self.navigator.gpu.requestAdapter() !== null;
    } catch (error) {
        return false;
    }
}

function requireGame() {
    if (!rustGame) {
        throw new Error('The Rust engine is not initialized');
    }
}

function parseJson(value) {
    return JSON.parse(value);
}

function errorMessage(error) {
    if (error instanceof Error) {
        return error.message;
    }
    return String(error);
}

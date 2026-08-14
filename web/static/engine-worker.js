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
            result = parseJson(rustGame.reset());
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

async function initialize({ simulations }) {
    const [modelResponse, metadataResponse] = await Promise.all([
        fetch('./model/champion.bin'),
        fetch('./model/metadata.json')
    ]);
    if (!modelResponse.ok) {
        throw new Error(`Model download failed (${modelResponse.status})`);
    }
    if (!metadataResponse.ok) {
        throw new Error(`Metadata download failed (${metadataResponse.status})`);
    }

    const modelBytes = new Uint8Array(await modelResponse.arrayBuffer());
    const metadataJson = await metadataResponse.text();
    const canUseWebGpu = 'gpu' in self.navigator;
    const attempts = canUseWebGpu ? ['webgpu', 'flex'] : ['flex'];
    const failures = [];

    for (const candidate of attempts) {
        try {
            const module = await import(`./pkg-${candidate}/yokai_web.js`);
            await module.default();
            rustGame = await module.createGame(modelBytes, metadataJson, simulations);
            backend = module.backendName();
            return {
                backend,
                generation: rustGame.generation,
                state: parseJson(rustGame.snapshotJson())
            };
        } catch (error) {
            failures.push(`${candidate}: ${errorMessage(error)}`);
        }
    }

    throw new Error(`No browser inference backend could start. ${failures.join(' / ')}`);
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

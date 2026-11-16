// Serves a directory of static files on this machine only, for the tests:
// node serve.js <directory> <port>
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';

const root = path.resolve(process.argv[2]);
const port = Number(process.argv[3]);

// A browser refuses a WebAssembly module or a script served under another
// content type.
const CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.png': 'image/png',
    '.jpg': 'image/jpeg'
};

const server = createServer(async (request, response) => {
    const { pathname } = new URL(request.url, 'http://localhost');
    const relative = decodeURIComponent(pathname).replace(/\/$/, '/index.html');
    const file = path.join(root, relative);
    try {
        // An address such as `/../secret` must not leave the directory.
        if (!file.startsWith(root + path.sep)) {
            throw new Error('outside the served directory');
        }
        const body = await readFile(file);
        response.writeHead(200, {
            'Content-Type': CONTENT_TYPES[path.extname(file)] ?? 'application/octet-stream',
            'Cache-Control': 'no-store'
        });
        response.end(body);
    } catch (error) {
        response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        response.end('Not found');
    }
});

server.listen(port, '127.0.0.1');

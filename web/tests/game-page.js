// Helpers shared by the browser tests: they open the built site, read the
// game that the page draws on its canvas and play it like a person would.
import { test as base, expect } from '@playwright/test';

export { expect };

// A canvas is a single picture: the browser cannot tell where a piece or a
// button is on it. The tests therefore read the positions from the Phaser
// objects of the page. The page keeps its game in a local variable, so this
// function, which the browser runs before the page's own scripts, wraps the
// constructor that Phaser exports and keeps the game that the page creates.
// The served files stay exactly as built.
function exposeGame() {
    let phaser;
    Object.defineProperty(window, 'Phaser', {
        configurable: true,
        get() {
            return phaser;
        },
        set(value) {
            phaser = value;
            const Game = value.Game;
            value.Game = class extends Game {
                constructor(configuration) {
                    super(configuration);
                    window.testedGame = this;
                }
            };
        }
    });
}

// Runs in the page and returns what a test needs to know about the game.
// This is the only place that names the fields of the scene in `main.js`.
// Positions are in page pixels, ready to be pressed.
function readGame() {
    const fatalError = document.getElementById('fatal-error');
    const view = {
        fatal: fatalError.hidden
            ? null
            : document.getElementById('fatal-error-message').textContent,
        state: null
    };
    const game = window.testedGame;
    const scene = game ? game.scene.getScene('game') : null;
    if (!scene || !scene.state) {
        return view;
    }

    const canvas = game.canvas.getBoundingClientRect();
    const toPage = (object) => ({
        x: canvas.x + object.x * canvas.width / game.config.width,
        y: canvas.y + object.y * canvas.height / game.config.height
    });
    const hand = (player) => scene.handSprites[player].map((entry) => ({
        kind: entry.kind,
        count: entry.count,
        countShown: Boolean(entry.label),
        ...toPage(entry.sprite)
    }));
    const pictureOf = (index) => {
        const sprite = scene.pieces.get(index);
        return sprite ? sprite.frame.name : null;
    };
    const newGame = scene.modal.list.find((object) => {
        return object.type === 'Text' && object.text === 'New game';
    });

    return {
        ...view,
        // The snapshot that the Rust engine sent to the page.
        state: scene.state,
        // Nothing moves any more: a person has to play, or the game is over.
        settled: !scene.busy && (scene.isHumanTurn() || scene.isModalOpen),
        resultShown: scene.isModalOpen,
        selected: scene.selected !== null,
        squares: scene.state.board.map((piece, index) => toPage(scene.squareCenter(index))),
        // Which picture of the sprite sheet each square shows.
        pictures: scene.state.board.map((piece, index) => pictureOf(index)),
        hands: { first: hand('first'), second: hand('second') },
        // The two faces offered when a move exists with and without promotion.
        promotionFaces: scene.promotionChoice
            ? scene.promotionChoice.list
                .filter((object) => object.type === 'Sprite')
                .map((sprite) => ({ picture: sprite.frame.name, ...toPage(sprite) }))
            : null,
        logo: toPage(scene.logo),
        newGameButton: toPage(newGame)
    };
}

// Rust lists a move that exists with and without promotion as two actions.
// A person chooses the move first and the promotion second, so the actions
// are grouped by move.
export function groupActions(actions) {
    const groups = new Map();
    for (const action of actions) {
        const key = action.type === 'move'
            ? `${action.from}-${action.to}`
            : `${action.piece}@${action.to}`;
        if (!groups.has(key)) {
            groups.set(key, []);
        }
        groups.get(key).push(action);
    }
    return [...groups.values()];
}

// A small random generator with a seed, so that a test replays the same
// choices at every run.
export function seededRandom(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6D2B79F5) >>> 0;
        let mixed = state;
        mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
        mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
        return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
}

export class GamePage {
    constructor(page, { hasTouch, baseURL }) {
        this.page = page;
        this.hasTouch = Boolean(hasTouch);
        this.origin = new URL(baseURL).origin;
        // Addresses that the page requested from the site.
        this.requests = [];
        // Anything a person would find in the browser's console or network
        // panel after a broken visit.
        this.problems = [];
    }

    async install() {
        const page = this.page;
        await page.addInitScript(exposeGame);
        page.on('request', (request) => {
            if (request.url().startsWith(this.origin)) {
                this.requests.push(new URL(request.url()));
            }
        });
        page.on('pageerror', (error) => {
            this.problems.push(`page error: ${error.message}`);
        });
        page.on('console', (message) => {
            if (message.type() === 'error') {
                this.problems.push(`console error: ${message.text()}`);
            }
        });
        page.on('requestfailed', (request) => {
            this.problems.push(`failed request: ${request.url()}`);
        });
        page.on('response', (response) => {
            if (response.status() >= 400) {
                this.problems.push(`status ${response.status()}: ${response.url()}`);
            }
        });
    }

    async read() {
        const view = await this.page.evaluate(readGame);
        if (view.fatal !== null) {
            throw new Error(`The page reported a fatal error: ${view.fatal}`);
        }
        return view;
    }

    // Reads the game until `condition` holds and returns that reading.
    async waitFor(condition, description, timeout = 120 * 1000) {
        const deadline = Date.now() + timeout;
        for (;;) {
            const view = await this.read();
            if (view.state && condition(view)) {
                return view;
            }
            if (Date.now() > deadline) {
                throw new Error(`Timed out waiting for ${description}`);
            }
            await this.page.waitForTimeout(50);
        }
    }

    async waitUntilSettled() {
        const view = await this.waitFor((reading) => reading.settled, 'the game to settle');
        // The loading screen fades out over the board.
        await expect(this.page.locator('#loading')).toHaveCSS('opacity', '0');
        return view;
    }

    async open() {
        await this.page.goto('./');
        return this.waitUntilSettled();
    }

    async reload() {
        await this.page.reload();
        return this.waitUntilSettled();
    }

    // A finger on a touch screen, a mouse click otherwise. Phaser makes a new
    // object answer presses when it draws its next image, a few milliseconds
    // after creating it. A person only presses what is on the screen, so the
    // test lets two images pass first.
    async press(point) {
        await this.page.evaluate(() => new Promise((resolve) => {
            requestAnimationFrame(() => requestAnimationFrame(resolve));
        }));
        if (this.hasTouch) {
            await this.page.touchscreen.tap(point.x, point.y);
        } else {
            await this.page.mouse.click(point.x, point.y);
        }
    }

    async pressElement(locator) {
        if (this.hasTouch) {
            await locator.tap();
        } else {
            await locator.click();
        }
    }

    // Chooses the number of players in the options panel, which must be open,
    // and starts the game.
    async startGame(players) {
        const form = this.page.locator('#menu-form');
        const choice = form.locator('label').filter({
            has: this.page.locator(`input[value="${players}"]`)
        });
        await this.pressElement(choice);
        await this.pressElement(form.locator('button[type="submit"]'));
        await expect(this.page.locator('#menu')).toBeHidden();
        return this.waitUntilSettled();
    }

    async openOptionsFromLogo() {
        const view = await this.read();
        await this.press(view.logo);
        await expect(this.page.locator('#menu')).toBeVisible();
    }

    // Presses "New game" on the panel that announces the result.
    async restartFromResult() {
        const view = await this.read();
        await this.press(view.newGameButton);
        return this.waitFor(
            (reading) => !reading.resultShown && reading.settled,
            'a new game after the result panel'
        );
    }

    // Plays one turn of the person to move. `choose` receives the reading of
    // the game and the legal moves grouped by `groupActions`; it returns the
    // group to play and, when the move exists with and without promotion,
    // whether to promote. Returns the action played and the readings taken
    // before the turn and once the game has settled again: in a one-player
    // game the champion has answered by then.
    async playTurn(choose) {
        const before = await this.read();
        const { group, promote } = choose(before, groupActions(before.state.legal_actions));
        let action = group[0];

        if (action.type === 'move') {
            await this.press(before.squares[action.from]);
        } else {
            const held = before.hands[before.state.side_to_move].find((entry) => {
                return entry.kind === action.piece;
            });
            await this.press(held);
        }
        await this.waitFor((reading) => reading.selected, 'the piece to be selected', 5000);
        await this.press(before.squares[action.to]);

        if (group.length > 1) {
            const offered = await this.waitFor(
                (reading) => reading.promotionFaces !== null,
                'the promotion choice',
                5000
            );
            // The face that shows the piece as it stands on the board keeps
            // it unpromoted; the other one promotes it.
            const current = before.pictures[action.from];
            const face = offered.promotionFaces.find((candidate) => {
                return (candidate.picture !== current) === promote;
            });
            await this.press(face);
            action = group.find((candidate) => candidate.promote === promote);
        }

        const after = await this.waitFor(
            (reading) => reading.settled && reading.state.ply > before.state.ply,
            `the action ${JSON.stringify(action)} to be played`
        );
        return { action, before, after };
    }
}

// Each test receives `game`, bound to its own page. Once the test body has
// run, the visit must have produced no error and no failed request.
export const test = base.extend({
    game: async ({ page, hasTouch, baseURL }, use) => {
        const game = new GamePage(page, { hasTouch, baseURL });
        await game.install();
        await use(game);
        expect(game.problems, 'errors and failed requests of the page').toEqual([]);
    }
});

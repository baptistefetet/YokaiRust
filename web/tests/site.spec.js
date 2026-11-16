// Scenarios played on the built site. Each one runs in the two projects of
// playwright.config.js.
import { expect, seededRandom, test } from './game-page.js';

const KING = 'koropokkuru';

function pick(items, random) {
    return items[Math.floor(random() * items.length)];
}

function capturedPiece(state, group) {
    return group[0].type === 'move' ? state.board[group[0].to] : null;
}

// Rows are numbered from the top of the screen: First, at the bottom, moves
// toward row 0 and Second toward the last row.
function advances(state, group) {
    const action = group[0];
    if (action.type !== 'move' || state.board[action.from].kind === KING) {
        return false;
    }
    const row = (index) => Math.floor(index / state.columns);
    return state.side_to_move === 'first'
        ? row(action.to) < row(action.from)
        : row(action.to) > row(action.from);
}

// A careless opponent for the champion: it captures when it can and moves at
// random otherwise.
function carelessPlayer(random) {
    return (view, groups) => {
        const captures = groups.filter((group) => capturedPiece(view.state, group));
        return { group: pick(captures.length > 0 ? captures : groups, random), promote: true };
    };
}

// Plays both sides of a two-player game so that it shows what `seen` still
// lacks, and ends the game once nothing is missing.
function curiousPlayer(random, seen) {
    return (view, groups) => {
        const state = view.state;
        const takesKing = (group) => capturedPiece(state, group)?.kind === KING;
        const everythingSeen = Object.values(seen).every(Boolean);

        const offered = groups.filter((group) => group.length > 1);
        if (offered.length > 0 && !(seen.promotion && seen.refusedPromotion)) {
            return { group: pick(offered, random), promote: !seen.promotion };
        }
        if (everythingSeen) {
            const wins = groups.filter(takesKing);
            if (wins.length > 0) {
                return { group: pick(wins, random), promote: true };
            }
        }
        // Hands fill up until one of them holds two pieces of a kind; only
        // then is a piece dropped.
        const drops = groups.filter((group) => group[0].type === 'drop');
        if (seen.pairInHand && !seen.drop && drops.length > 0) {
            return { group: pick(drops, random), promote: true };
        }
        // Until everything has been seen, the Koropokkuru is left alone so
        // that the game goes on.
        const moves = groups.filter((group) => {
            return group[0].type === 'move' && (everythingSeen || !takesKing(group));
        });
        const captures = moves.filter((group) => capturedPiece(state, group));
        if (captures.length > 0) {
            return { group: pick(captures, random), promote: true };
        }
        const forward = moves.filter((group) => advances(state, group));
        if (forward.length > 0 && random() < 0.75) {
            return { group: pick(forward, random), promote: true };
        }
        return { group: pick(moves.length > 0 ? moves : groups, random), promote: true };
    };
}

test('loads the files of one build and the expected engine module', async ({ game, page }, testInfo) => {
    await game.open();

    const modules = new Set(
        game.requests
            .map((url) => url.pathname.match(/\/pkg-([a-z]+)\//))
            .filter(Boolean)
            .map((match) => match[1])
    );
    expect([...modules]).toEqual([testInfo.project.metadata.backend]);

    // Every file except the page and its icon carries the build identifier,
    // so a browser cannot combine files kept from two builds.
    const stylesheet = await page.locator('link[rel="stylesheet"]').getAttribute('href');
    const buildQuery = new URL(stylesheet, page.url()).search;
    expect(buildQuery).toMatch(/^\?v=.+/);
    const unmarked = game.requests
        .filter((url) => url.pathname !== '/' && !url.pathname.endsWith('/favicon.png'))
        .filter((url) => url.search !== buildQuery)
        .map((url) => url.pathname + url.search);
    expect(unmarked).toEqual([]);
});

test('one player meets the champion until the game ends, then starts again', async ({ game }) => {
    await game.open();
    await game.startGame(1);

    const choose = carelessPlayer(seededRandom(20261116));
    const first = await game.playTurn(choose);
    // The champion has answered the person's move.
    expect(first.after.state.ply).toBe(first.before.state.ply + 2);

    let view = first.after;
    while (!view.resultShown) {
        view = (await game.playTurn(choose)).after;
    }
    expect(view.state.outcome.status).not.toBe('ongoing');

    view = await game.restartFromResult();
    expect(view.state.outcome.status).toBe('ongoing');
    expect(view.state.ply).toBeLessThanOrEqual(1);
});

test('two players capture, drop and choose promotions until the game ends', async ({ game, page }) => {
    await game.open();
    let view = await game.startGame(2);
    expect(view.state.humans).toHaveLength(2);

    const seen = {
        capture: false,
        drop: false,
        promotion: false,
        refusedPromotion: false,
        pairInHand: false
    };
    const choose = curiousPlayer(seededRandom(20261117), seen);
    const everythingSeen = () => Object.values(seen).every(Boolean);

    // One game normally shows everything. A game that ends too early is
    // followed by another one from the result panel.
    for (let games = 1; ; games += 1) {
        while (!view.resultShown) {
            const turn = await game.playTurn(choose);
            const { action, before } = turn;
            view = turn.after;
            const mover = before.state.side_to_move;

            if (action.type === 'drop') {
                seen.drop = true;
                expect(view.state.board[action.to]).toEqual({ kind: action.piece, owner: mover });
            } else {
                const moved = before.state.board[action.from];
                const offered = before.state.legal_actions.filter((candidate) => {
                    return candidate.type === 'move'
                        && candidate.from === action.from
                        && candidate.to === action.to;
                }).length > 1;
                if (capturedPiece(before.state, [action])) {
                    seen.capture = true;
                }
                if (offered && action.promote) {
                    seen.promotion = true;
                    expect(view.state.board[action.to].kind).not.toBe(moved.kind);
                }
                if (offered && !action.promote) {
                    seen.refusedPromotion = true;
                    expect(view.state.board[action.to].kind).toBe(moved.kind);
                }
            }
            for (const held of [...view.hands.first, ...view.hands.second]) {
                if (held.count > 1) {
                    seen.pairInHand = true;
                    // A stack shows how many pieces it holds.
                    expect(held.countShown).toBe(true);
                }
            }
        }
        expect(view.state.outcome.status).not.toBe('ongoing');
        if (everythingSeen()) {
            break;
        }
        expect(games, `still missing: ${JSON.stringify(seen)}`).toBeLessThan(4);
        view = await game.restartFromResult();
    }

    view = await game.restartFromResult();
    expect(view.state.ply).toBe(0);

    // The options open from the logo and close without touching the game.
    await game.openOptionsFromLogo();
    await game.pressElement(page.locator('#menu-cancel'));
    await expect(page.locator('#menu')).toBeHidden();

    // The choice of two players is kept by the browser across a reload.
    view = await game.reload();
    expect(view.state.humans).toHaveLength(2);
    await expect(page.locator('#menu')).toBeHidden();
});

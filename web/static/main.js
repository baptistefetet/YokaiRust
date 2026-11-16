// Pixel measurements of the 600×900 board image. The grid of squares is
// divided by the board dimensions that Rust reports in each snapshot.
const LAYOUT = {
    gameWidth: 600,
    gameHeight: 900,
    boardX: 26,
    boardY: 104,
    boardWidth: 547,
    boardHeight: 689,
    characterSize: 128,
    characterScale: 0.7,
    handScale: 0.5,
    handOffset: 18,
    tweenMs: 420
};

const PIECE_FRAME = {
    koropokkuru: 0,
    kirin: 1,
    oni: 2,
    super_oni: 3,
    kodama: 4,
    kodama_samurai: 5
};

const DEFAULT_SIMULATIONS = 200;
const MINIMUM_THINK_MS = 260;
const SETTINGS_KEY = 'yokai-no-mori.settings';
const TEXT_FONT = 'system-ui, sans-serif';

// The only stored choice is the number of human players. Storage can be
// unavailable (private browsing, embedded pages); the game then starts with
// the default every time.
function loadSettings() {
    try {
        const stored = JSON.parse(window.localStorage.getItem(SETTINGS_KEY));
        if (stored && (stored.players === 1 || stored.players === 2)) {
            return { players: stored.players, isFirstVisit: false };
        }
    } catch (error) {
        // Fall through to the default.
    }
    return { players: 1, isFirstVisit: true };
}

function saveSettings(settings) {
    try {
        window.localStorage.setItem(SETTINGS_KEY, JSON.stringify({ players: settings.players }));
    } catch (error) {
        // The choice still applies to the current page.
    }
}

class EngineClient {
    constructor() {
        this.worker = new Worker('./engine-worker.js', { type: 'module' });
        this.nextRequestId = 1;
        this.pending = new Map();
        this.worker.addEventListener('message', (event) => {
            const request = this.pending.get(event.data.id);
            if (!request) {
                return;
            }
            this.pending.delete(event.data.id);
            if (event.data.error) {
                request.reject(new Error(event.data.error));
            } else {
                request.resolve(event.data.result);
            }
        });
        this.worker.addEventListener('error', (event) => {
            const error = new Error(event.message || 'The engine worker stopped unexpectedly');
            for (const request of this.pending.values()) {
                request.reject(error);
            }
            this.pending.clear();
        });
    }

    initialize(simulations, humanPlayers) {
        return this.request('init', { simulations, humanPlayers });
    }

    reset(humanPlayers) {
        return this.request('reset', { humanPlayers });
    }

    playHuman(action) {
        return this.request('human', { action });
    }

    playAi() {
        return this.request('ai');
    }

    request(command, payload = {}) {
        const id = this.nextRequestId;
        this.nextRequestId += 1;
        return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.worker.postMessage({ id, command, payload });
        });
    }
}

class GameScene extends Phaser.Scene {
    constructor() {
        super('game');
    }

    preload() {
        this.load.image('board', './assets/images/board.jpg');
        this.load.spritesheet('characters', './assets/images/characters.png', {
            frameWidth: LAYOUT.characterSize,
            frameHeight: LAYOUT.characterSize
        });
        this.load.image('logo', './assets/images/logo.png');
    }

    async create() {
        this.margin = 0;
        this.state = null;
        this.settings = loadSettings();
        this.busy = true;
        this.selected = null;
        this.promotionChoice = null;
        this.isModalOpen = false;
        this.pieces = new Map();
        this.handSprites = { first: [], second: [] };
        // Every piece and hand count belongs to this group, including a piece
        // that only exists while it travels from a hand stack to the board.
        // Redrawing empties the group, so no sprite outlives its position.
        this.pieceObjects = this.add.group();
        this.highlights = [];
        this.pulseTweens = [];
        this.input.setTopOnly(true);

        const boardImage = this.textures.get('board').getSourceImage();
        this.imageWidth = boardImage.width;
        this.imageHeight = boardImage.height;
        this.scaleX = this.imageWidth / LAYOUT.gameWidth;
        this.scaleY = this.imageHeight / LAYOUT.gameHeight;

        this.add.image(
            this.margin + this.imageWidth / 2,
            this.margin + this.imageHeight / 2,
            'board'
        );
        this.cameras.main.setBackgroundColor('#ffffff');
        this.placeAbout();
        this.scale.on('resize', () => {
            this.placeAbout();
        });
        this.createLogo();
        this.createGameOverModal();
        this.bindMenu();
        this.input.on('pointerdown', (pointer) => {
            this.handleBoardPointer(pointer);
        });

        this.engine = new EngineClient();
        try {
            const initialized = await this.engine.initialize(
                DEFAULT_SIMULATIONS,
                this.settings.players
            );
            this.state = initialized.state;
            this.drawAll();
            this.finishLoading();
            await this.continueGame();
            if (this.settings.isFirstVisit) {
                this.openMenu();
            }
        } catch (error) {
            this.showFatalError(error);
        }
    }

    // The version and source link is an HTML element above the canvas. Phaser
    // scales the canvas and centres it in the window, so the link follows the
    // bottom-right corner of the board image each time the canvas moves.
    placeAbout() {
        const about = document.getElementById('about');
        const page = document.getElementById('app').getBoundingClientRect();
        const board = this.game.canvas.getBoundingClientRect();
        about.style.right = `${page.right - board.right}px`;
        about.style.bottom = `${page.bottom - board.bottom}px`;
    }

    squareWidth() {
        return LAYOUT.boardWidth * this.scaleX / this.state.columns;
    }

    squareHeight() {
        return LAYOUT.boardHeight * this.scaleY / this.state.rows;
    }

    toPixel(column, row) {
        return {
            x: this.margin + LAYOUT.boardX * this.scaleX + (column + 0.5) * this.squareWidth(),
            y: this.margin + LAYOUT.boardY * this.scaleY + (row + 0.5) * this.squareHeight()
        };
    }

    squareCenter(index) {
        return this.toPixel(index % this.state.columns, Math.floor(index / this.state.columns));
    }

    fromPixel(x, y) {
        const localX = x - (this.margin + LAYOUT.boardX * this.scaleX);
        const localY = y - (this.margin + LAYOUT.boardY * this.scaleY);
        if (
            localX < 0
            || localY < 0
            || localX >= LAYOUT.boardWidth * this.scaleX
            || localY >= LAYOUT.boardHeight * this.scaleY
        ) {
            return null;
        }
        const column = Math.floor(localX / this.squareWidth());
        const row = Math.floor(localY / this.squareHeight());
        if (column >= this.state.columns || row >= this.state.rows) {
            return null;
        }
        return { column, row, index: row * this.state.columns + column };
    }

    // A hand shows one piece per kind, with a count when the player holds
    // several. Each kind keeps its own slot; the Kodama, captured most often,
    // has the slot nearest to the corner.
    handSlot(kindIndex) {
        return this.state.hand_kinds.length - 1 - kindIndex;
    }

    handCenter(player, index) {
        if (player === 'first') {
            return {
                x: this.margin + LAYOUT.handOffset * this.scaleX
                    + LAYOUT.handScale * this.scaleX
                    * (index * LAYOUT.characterSize + LAYOUT.characterSize / 2),
                y: this.margin + this.imageHeight - LAYOUT.handOffset * this.scaleY
                    - LAYOUT.handScale * this.scaleY * LAYOUT.characterSize / 2
            };
        }
        return {
            x: this.margin + this.imageWidth - LAYOUT.handOffset * this.scaleX
                - LAYOUT.handScale * this.scaleX
                * (index * LAYOUT.characterSize + LAYOUT.characterSize / 2),
            y: this.margin + LAYOUT.handOffset * this.scaleY
                + LAYOUT.handScale * this.scaleY * LAYOUT.characterSize / 2
        };
    }

    createLogo() {
        const size = 48 * Math.min(this.scaleX, this.scaleY);
        const x = this.margin + 9 * this.scaleX + size / 2;
        const y = this.margin + 11 * this.scaleY + size / 2;
        this.logo = this.add.image(x, y, 'logo')
            .setDisplaySize(size, size)
            .setDepth(100)
            .setInteractive({ useHandCursor: true });
        this.logo.on('pointerdown', (pointer, localX, localY, event) => {
            if (event && event.stopPropagation) {
                event.stopPropagation();
            }
            if (!this.busy) {
                this.openMenu();
            }
        });
        this.logoSpin = this.tweens.add({
            targets: this.logo,
            angle: '+=360',
            duration: 1000,
            repeat: -1,
            ease: 'Linear',
            paused: true
        });
    }

    // Input stays locked while an action is animated and while the champion
    // searches; the logo only spins during the search.
    lockInput() {
        this.busy = true;
        this.stopPulseAnimations();
        this.closePromotionChoice();
        this.clearHighlights();
        this.selected = null;
        this.logo.disableInteractive();
    }

    unlockInput() {
        this.busy = false;
        this.logoSpin.pause();
        this.logo.setAngle(0);
        this.logo.setInteractive({ useHandCursor: true });
    }

    spinLogo() {
        if (this.logoSpin.isPaused()) {
            this.logoSpin.resume();
        }
    }

    isHumanSide(player) {
        return this.state.humans.includes(player);
    }

    makeBoardPiece(index, piece) {
        const position = this.squareCenter(index);
        const sprite = this.add.sprite(
            position.x,
            position.y,
            'characters',
            PIECE_FRAME[piece.kind]
        );
        sprite.setScale(
            LAYOUT.characterScale * this.scaleX,
            LAYOUT.characterScale * this.scaleY
        );
        if (piece.owner === 'second') {
            sprite.setAngle(180);
        }
        sprite.setDepth(10);
        this.pieceObjects.add(sprite);
        return sprite;
    }

    makeHandPiece(player, index, kind) {
        const position = this.handCenter(player, index);
        const sprite = this.add.sprite(
            position.x,
            position.y,
            'characters',
            PIECE_FRAME[kind]
        );
        sprite.setScale(
            LAYOUT.handScale * this.scaleX,
            LAYOUT.handScale * this.scaleY
        );
        if (player === 'second') {
            sprite.setAngle(180);
        }
        sprite.setDepth(5);
        this.pieceObjects.add(sprite);
        return sprite;
    }

    makeHandCount(player, index, count) {
        const center = this.handCenter(player, index);
        const reach = LAYOUT.handScale * LAYOUT.characterSize / 2 - 6;
        // The count sits on the corner nearest to the piece's owner.
        const direction = player === 'second' ? -1 : 1;
        const label = this.add.text(
            center.x + direction * reach * this.scaleX,
            center.y + direction * reach * this.scaleY,
            `×${count}`,
            {
                fontFamily: TEXT_FONT,
                fontSize: '17px',
                fontStyle: 'bold',
                color: '#ffffff',
                stroke: '#27231f',
                strokeThickness: 4
            }
        ).setOrigin(0.5).setDepth(6);
        // A second person reads their hand from the other side of the screen.
        if (player === 'second' && this.isHumanSide('second')) {
            label.setAngle(180);
        }
        this.pieceObjects.add(label);
        return label;
    }

    drawAll() {
        this.stopPulseAnimations();
        this.pieceObjects.clear(true, true);
        this.pieces.clear();
        this.handSprites = { first: [], second: [] };

        const mover = this.state.side_to_move;
        this.state.board.forEach((piece, index) => {
            if (!piece) {
                return;
            }
            const sprite = this.makeBoardPiece(index, piece);
            const canSelect = this.isHumanTurn()
                && piece.owner === mover
                && this.moveActionsFrom(index).length > 0;
            if (canSelect) {
                sprite.setInteractive({ useHandCursor: true });
            }
            this.pieces.set(index, sprite);
        });

        for (const player of ['first', 'second']) {
            this.handCounts(this.state, player).forEach((count, index) => {
                if (count === 0) {
                    return;
                }
                const kind = this.state.hand_kinds[index];
                const slot = this.handSlot(index);
                const sprite = this.makeHandPiece(player, slot, kind);
                const label = count > 1 ? this.makeHandCount(player, slot, count) : null;
                const canDrop = this.isHumanTurn()
                    && player === mover
                    && this.dropActionsFor(kind).length > 0;
                if (canDrop) {
                    sprite.setInteractive({ useHandCursor: true });
                    sprite.on('pointerdown', (pointer, localX, localY, event) => {
                        if (event && event.stopPropagation) {
                            event.stopPropagation();
                        }
                        this.selectHandPiece(kind);
                    });
                }
                this.handSprites[player].push({ kind, slot, count, sprite, label });
            });
        }
        this.startPulseAnimations();
    }

    handCounts(state, player) {
        return state.hands[player === 'first' ? 0 : 1];
    }

    isHumanTurn() {
        return Boolean(
            this.state
            && !this.busy
            && !this.isModalOpen
            && this.state.outcome.status === 'ongoing'
            && this.isHumanSide(this.state.side_to_move)
        );
    }

    moveActionsFrom(index) {
        return this.state.legal_actions.filter((action) => {
            return action.type === 'move' && action.from === index;
        });
    }

    dropActionsFor(kind) {
        return this.state.legal_actions.filter((action) => {
            return action.type === 'drop' && action.piece === kind;
        });
    }

    handleBoardPointer(pointer) {
        if (!this.isHumanTurn()) {
            return;
        }
        if (this.promotionChoice) {
            // A press outside the two proposed pieces cancels the move.
            this.clearSelection();
            return;
        }
        const square = this.fromPixel(pointer.worldX, pointer.worldY);
        if (!square) {
            this.clearSelection();
            return;
        }

        if (this.selected) {
            const actions = this.selected.actions.filter((candidate) => {
                return candidate.to === square.index;
            });
            if (actions.length === 1) {
                this.playHumanAction(actions[0]);
                return;
            }
            if (actions.length > 1) {
                // The same move exists with and without promotion.
                this.openPromotionChoice(actions);
                return;
            }
        }

        const piece = this.state.board[square.index];
        if (piece && piece.owner === this.state.side_to_move) {
            const actions = this.moveActionsFrom(square.index);
            if (actions.length > 0) {
                this.selectBoardPiece(square.index, actions);
                return;
            }
        }
        this.clearSelection();
    }

    selectBoardPiece(index, actions) {
        this.stopPulseAnimations();
        this.clearHighlights();
        this.clearHandTints();
        this.selected = { type: 'board', index, actions };
        const sprite = this.pieces.get(index);
        if (sprite) {
            sprite.setTint(0xffe066);
        }
        this.showActionTargets(actions, 0x00bcd4);
    }

    selectHandPiece(kind) {
        if (!this.isHumanTurn()) {
            return;
        }
        this.closePromotionChoice();
        if (this.selected && this.selected.type === 'drop' && this.selected.kind === kind) {
            this.clearSelection();
            return;
        }
        const actions = this.dropActionsFor(kind);
        if (actions.length === 0) {
            return;
        }
        this.stopPulseAnimations();
        this.clearHighlights();
        this.clearBoardTints();
        this.clearHandTints();
        this.selected = { type: 'drop', kind, actions };
        for (const entry of this.handSprites[this.state.side_to_move]) {
            if (entry.kind === kind) {
                entry.sprite.setTint(0xffe066);
            }
        }
        this.showActionTargets(actions, 0xffc107);
    }

    showActionTargets(actions, color) {
        const targets = new Set(actions.map((action) => action.to));
        for (const target of targets) {
            const center = this.squareCenter(target);
            const highlight = this.add.rectangle(
                center.x,
                center.y,
                this.squareWidth() * 0.9,
                this.squareHeight() * 0.9,
                color,
                0.25
            ).setStrokeStyle(2, color, 0.9);
            this.highlights.push(highlight);
        }
    }

    // Shows the moving piece's two faces over the target square. Rust lists
    // the plain and the promoting move as two legal actions; the face that
    // the player presses selects one of them.
    openPromotionChoice(actions) {
        const mover = this.state.side_to_move;
        const kind = this.state.board[actions[0].from].kind;
        const promotion = this.state.promotions.find((entry) => entry.kind === kind);
        const choices = [
            { action: actions.find((action) => action.promote), kind: promotion.promoted },
            { action: actions.find((action) => !action.promote), kind }
        ];
        if (mover === 'second') {
            // Second sees the screen from the other side.
            choices.reverse();
        }

        const size = LAYOUT.characterSize * LAYOUT.characterScale * this.scaleX;
        const gap = 10;
        const padding = 10;
        const width = 2 * size + gap + 2 * padding;
        const height = size + 2 * padding;
        const target = this.squareCenter(actions[0].to);
        const centerX = Phaser.Math.Clamp(
            target.x,
            this.margin + width / 2 + 4,
            this.margin + this.imageWidth - width / 2 - 4
        );
        const centerY = target.y;

        const shade = this.add.rectangle(
            this.margin,
            this.margin,
            this.imageWidth,
            this.imageHeight,
            0x000000,
            0.3
        ).setOrigin(0);
        const panel = this.add.graphics();
        panel.fillStyle(0xffffff, 1);
        panel.lineStyle(3, 0x222222, 1);
        panel.fillRoundedRect(centerX - width / 2, centerY - height / 2, width, height, 14);
        panel.strokeRoundedRect(centerX - width / 2, centerY - height / 2, width, height, 14);

        const container = this.add.container(0, 0, [shade, panel]).setDepth(150);
        choices.forEach((choice, index) => {
            const x = centerX + (index === 0 ? -1 : 1) * (size + gap) / 2;
            const sprite = this.add.sprite(x, centerY, 'characters', PIECE_FRAME[choice.kind]);
            sprite.setScale(
                LAYOUT.characterScale * this.scaleX,
                LAYOUT.characterScale * this.scaleY
            );
            if (mover === 'second') {
                sprite.setAngle(180);
            }
            sprite.setInteractive({ useHandCursor: true });
            sprite.on('pointerdown', (pointer, localX, localY, event) => {
                if (event && event.stopPropagation) {
                    event.stopPropagation();
                }
                this.playHumanAction(choice.action);
            });
            container.add(sprite);
        });
        this.promotionChoice = container;
    }

    closePromotionChoice() {
        if (this.promotionChoice) {
            this.promotionChoice.destroy();
            this.promotionChoice = null;
        }
    }

    clearSelection() {
        this.selected = null;
        this.closePromotionChoice();
        this.clearHighlights();
        this.clearBoardTints();
        this.clearHandTints();
        this.startPulseAnimations();
    }

    clearHighlights() {
        for (const highlight of this.highlights) {
            highlight.destroy();
        }
        this.highlights = [];
    }

    clearBoardTints() {
        for (const sprite of this.pieces.values()) {
            sprite.clearTint();
        }
    }

    clearHandTints() {
        for (const player of ['first', 'second']) {
            for (const entry of this.handSprites[player]) {
                entry.sprite.clearTint();
            }
        }
    }

    // Starts from a clean state: a tween left running would be joined by a
    // second one that takes the sprite's inflated size as its resting size.
    startPulseAnimations() {
        this.stopPulseAnimations();
        if (!this.isHumanTurn() || this.selected) {
            return;
        }
        const mover = this.state.side_to_move;
        this.state.board.forEach((piece, index) => {
            if (piece && piece.owner === mover && this.moveActionsFrom(index).length > 0) {
                const sprite = this.pieces.get(index);
                if (sprite) {
                    this.addPulseAnimation(sprite);
                }
            }
        });
        for (const entry of this.handSprites[mover]) {
            if (this.dropActionsFor(entry.kind).length > 0) {
                this.addPulseAnimation(entry.sprite);
            }
        }
    }

    addPulseAnimation(sprite) {
        const scaleX = sprite.scaleX;
        const scaleY = sprite.scaleY;
        const tween = this.tweens.add({
            targets: sprite,
            scaleX: scaleX * 1.04,
            scaleY: scaleY * 1.04,
            duration: 500,
            ease: 'Sine.easeInOut',
            yoyo: true,
            repeat: -1
        });
        this.pulseTweens.push({ tween, sprite, scaleX, scaleY });
    }

    stopPulseAnimations() {
        for (const entry of this.pulseTweens) {
            entry.tween.remove();
            if (entry.sprite.scene) {
                entry.sprite.setScale(entry.scaleX, entry.scaleY);
            }
        }
        this.pulseTweens = [];
    }

    async playHumanAction(action) {
        if (!this.isHumanTurn()) {
            return;
        }
        const previousState = this.state;
        this.lockInput();
        try {
            const response = await this.engine.playHuman(action);
            await this.animateAppliedAction(previousState, response);
            this.state = response.state;
            this.drawAll();
            await this.continueGame();
        } catch (error) {
            this.unlockInput();
            this.showFatalError(error);
        }
    }

    // Lets the champion play until a person has to move or the game ends,
    // then hands the board back.
    async continueGame() {
        this.lockInput();
        while (
            this.state.outcome.status === 'ongoing'
            && !this.isHumanSide(this.state.side_to_move)
        ) {
            await this.playAiMove();
        }
        this.unlockInput();
        this.drawAll();
        this.showOutcomeIfFinished();
    }

    async playAiMove() {
        const previousState = this.state;
        const started = performance.now();
        this.spinLogo();
        const response = await this.engine.playAi();
        const elapsed = performance.now() - started;
        if (elapsed < MINIMUM_THINK_MS) {
            await sleep(MINIMUM_THINK_MS - elapsed);
        }
        await this.animateAppliedAction(previousState, response);
        this.state = response.state;
        this.drawAll();
    }

    async animateAppliedAction(previousState, response) {
        this.stopPulseAnimations();
        const action = response.action;
        const player = previousState.side_to_move;
        const destination = this.squareCenter(action.to);
        const capturedSprite = this.pieces.get(action.to) || null;
        let movingSprite;

        if (action.type === 'move') {
            movingSprite = this.pieces.get(action.from);
        } else {
            const handEntry = this.handSprites[player].find((entry) => {
                return entry.kind === action.piece;
            });
            if (handEntry.count > 1) {
                // One piece leaves the stack; the others stay in the hand.
                movingSprite = this.makeHandPiece(player, handEntry.slot, action.piece);
                const remaining = handEntry.count - 1;
                handEntry.label.setText(`×${remaining}`).setVisible(remaining > 1);
            } else {
                movingSprite = handEntry.sprite;
            }
        }
        movingSprite.setDepth(60);

        await this.tweenPromise({
            targets: movingSprite,
            x: destination.x,
            y: destination.y,
            scaleX: LAYOUT.characterScale * this.scaleX,
            scaleY: LAYOUT.characterScale * this.scaleY,
            duration: LAYOUT.tweenMs,
            ease: 'Sine.easeInOut'
        });

        if (response.captured && capturedSprite) {
            // The hand column that grew tells where the captured piece goes
            // and which face it shows there: a promoted piece is captured as
            // its plain kind. Capturing the Koropokkuru fills no column.
            const before = this.handCounts(previousState, player);
            const after = this.handCounts(response.state, player);
            const handIndex = after.findIndex((count, index) => count > before[index]);
            capturedSprite.setDepth(55);
            if (handIndex >= 0) {
                capturedSprite.setFrame(PIECE_FRAME[response.state.hand_kinds[handIndex]]);
                const handDestination = this.handCenter(player, this.handSlot(handIndex));
                await this.tweenPromise({
                    targets: capturedSprite,
                    x: handDestination.x,
                    y: handDestination.y,
                    angle: player === 'second' ? 180 : 0,
                    scaleX: LAYOUT.handScale * this.scaleX,
                    scaleY: LAYOUT.handScale * this.scaleY,
                    duration: Math.max(330, Math.floor(LAYOUT.tweenMs * 0.9)),
                    ease: 'Sine.easeInOut'
                });
            } else {
                await this.tweenPromise({
                    targets: capturedSprite,
                    alpha: 0,
                    scaleX: 0.25,
                    scaleY: 0.25,
                    duration: 260,
                    ease: 'Quad.easeIn'
                });
            }
        }

        if (response.promoted) {
            movingSprite.setFrame(PIECE_FRAME[response.state.board[action.to].kind]);
            movingSprite.setTint(0xffffaa);
            const baseScaleX = LAYOUT.characterScale * this.scaleX;
            const baseScaleY = LAYOUT.characterScale * this.scaleY;
            await this.tweenPromise({
                targets: movingSprite,
                scaleX: baseScaleX * 1.12,
                scaleY: baseScaleY * 1.12,
                duration: 120,
                ease: 'Quad.easeOut',
                yoyo: true
            });
            movingSprite.clearTint();
        }
    }

    tweenPromise(configuration) {
        return new Promise((resolve) => {
            this.tweens.add({ ...configuration, onComplete: resolve });
        });
    }

    showOutcomeIfFinished() {
        const outcome = this.state.outcome;
        if (outcome.status === 'ongoing') {
            return false;
        }
        // The reason sentence is rendered by Rust (`outcome_text`), so this
        // file never depends on the serialized names of outcome reasons.
        const reason = this.state.outcome_text || 'The game is over.';
        if (outcome.status === 'draw') {
            this.showGameOver('Draw', reason);
        } else if (this.state.humans.length > 1) {
            const winner = outcome.player === 'first' ? 'Bottom' : 'Top';
            this.showGameOver(`${winner} player wins!`, reason);
        } else if (this.isHumanSide(outcome.player)) {
            this.showGameOver('Victory!', reason);
        } else {
            this.showGameOver('Defeat', reason);
        }
        return true;
    }

    createGameOverModal() {
        this.modal = this.add.container(0, 0).setDepth(200).setVisible(false);
        const blocker = this.add.rectangle(
            this.margin,
            this.margin,
            this.imageWidth,
            this.imageHeight,
            0x000000,
            0.45
        ).setOrigin(0).setInteractive();
        const panelWidth = Math.min(this.imageWidth * 0.72, 480);
        const panelHeight = 240;
        const centerX = this.margin + this.imageWidth / 2;
        const centerY = this.margin + this.imageHeight / 2;
        const panel = this.add.graphics();
        panel.fillStyle(0xffffff, 1);
        panel.lineStyle(3, 0x222222, 1);
        panel.fillRoundedRect(
            centerX - panelWidth / 2,
            centerY - panelHeight / 2,
            panelWidth,
            panelHeight,
            18
        );
        panel.strokeRoundedRect(
            centerX - panelWidth / 2,
            centerY - panelHeight / 2,
            panelWidth,
            panelHeight,
            18
        );
        this.modalTitle = this.add.text(centerX, centerY - 50, 'Victory!', {
            fontFamily: TEXT_FONT,
            fontSize: '36px',
            color: '#111111',
            fontStyle: 'bold'
        }).setOrigin(0.5);
        this.modalSubtitle = this.add.text(centerX, centerY - 8, '', {
            fontFamily: TEXT_FONT,
            fontSize: '16px',
            color: '#444444',
            align: 'center',
            wordWrap: { width: panelWidth - 42 }
        }).setOrigin(0.5);
        const buttons = [
            this.createModalButton(centerX - 96, centerY + 65, 'New game', 0x222222, () => {
                this.restartGame();
            }),
            this.createModalButton(centerX + 96, centerY + 65, 'Options', 0x6b635a, () => {
                this.openMenu();
            })
        ];
        this.modal.add([blocker, panel, this.modalTitle, this.modalSubtitle, ...buttons.flat()]);
    }

    createModalButton(centerX, centerY, text, color, onPress) {
        const width = 172;
        const height = 48;
        const background = this.add.graphics();
        background.fillStyle(color, 1);
        background.fillRoundedRect(centerX - width / 2, centerY - height / 2, width, height, 12);
        const hitArea = this.add.rectangle(centerX, centerY, width, height, 0x000000, 0)
            .setInteractive({ useHandCursor: true });
        const label = this.add.text(centerX, centerY, text, {
            fontFamily: TEXT_FONT,
            fontSize: '18px',
            color: '#ffffff'
        }).setOrigin(0.5);
        hitArea.on('pointerover', () => {
            label.setScale(1.05);
        });
        hitArea.on('pointerout', () => {
            label.setScale(1);
        });
        hitArea.on('pointerdown', (pointer, localX, localY, event) => {
            if (event && event.stopPropagation) {
                event.stopPropagation();
            }
            onPress();
        });
        return [background, hitArea, label];
    }

    showGameOver(title, subtitle) {
        this.isModalOpen = true;
        this.stopPulseAnimations();
        this.modalTitle.setText(title);
        this.modalSubtitle.setText(subtitle);
        this.modal.setVisible(true);
    }

    async restartGame() {
        if (this.busy) {
            return;
        }
        this.isModalOpen = false;
        this.modal.setVisible(false);
        this.lockInput();
        try {
            this.state = await this.engine.reset(this.settings.players);
            this.drawAll();
            await this.continueGame();
        } catch (error) {
            this.unlockInput();
            this.showFatalError(error);
        }
    }

    // The options are a plain HTML form above the canvas: it stays readable
    // at any canvas scale and works with a keyboard.
    bindMenu() {
        this.menu = document.getElementById('menu');
        const form = document.getElementById('menu-form');
        form.addEventListener('submit', (event) => {
            event.preventDefault();
            const players = Number(new FormData(form).get('players'));
            this.settings = { players: players === 2 ? 2 : 1, isFirstVisit: false };
            saveSettings(this.settings);
            this.menu.hidden = true;
            this.restartGame();
        });
        document.getElementById('menu-cancel').addEventListener('click', () => {
            this.menu.hidden = true;
        });
    }

    openMenu() {
        const form = document.getElementById('menu-form');
        form.elements.players.value = String(this.settings.players);
        this.menu.hidden = false;
        form.querySelector('input:checked').focus();
    }

    finishLoading() {
        const title = document.getElementById('loading-title');
        const detail = document.getElementById('loading-detail');
        const loading = document.getElementById('loading');
        title.textContent = 'Ready!';
        detail.textContent = 'Your opponent is ready.';
        window.setTimeout(() => {
            loading.classList.add('ready');
        }, 220);
        window.setTimeout(() => {
            loading.hidden = true;
        }, 520);
    }

    showFatalError(error) {
        console.error(error);
        const loading = document.getElementById('loading');
        const panel = document.getElementById('fatal-error');
        loading.hidden = true;
        document.getElementById('fatal-error-message').textContent = error.message || String(error);
        panel.hidden = false;
    }
}

function sleep(milliseconds) {
    return new Promise((resolve) => {
        window.setTimeout(resolve, milliseconds);
    });
}

document.getElementById('reload-button').addEventListener('click', () => {
    window.location.reload();
});

new Phaser.Game({
    type: Phaser.AUTO,
    parent: 'game',
    width: LAYOUT.gameWidth,
    height: LAYOUT.gameHeight,
    scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH
    },
    scene: [GameScene],
    backgroundColor: '#ffffff'
});

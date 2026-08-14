const LAYOUT = {
    gameWidth: 600,
    gameHeight: 900,
    boardX: 61,
    boardY: 130,
    boardWidth: 476,
    boardHeight: 638,
    columns: 3,
    rows: 4,
    squareWidth: 159,
    squareHeight: 159,
    characterSize: 128,
    characterScale: 1,
    handScale: 0.7,
    handOffset: 18,
    tweenMs: 420
};

const PIECE_FRAME = {
    koropokkuru: 0,
    tanuki: 1,
    kitsune: 2,
    kodama: 3,
    kodama_samurai: 4
};

const HAND_KINDS = ['tanuki', 'kitsune', 'kodama'];
const DEFAULT_SIMULATIONS = 200;
const MINIMUM_THINK_MS = 260;

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

    initialize(simulations) {
        return this.request('init', { simulations });
    }

    reset() {
        return this.request('reset');
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
        this.load.image('board', './assets/images/3x4/board.jpg');
        this.load.spritesheet('characters', './assets/images/3x4/characters.png', {
            frameWidth: LAYOUT.characterSize,
            frameHeight: LAYOUT.characterSize
        });
        this.load.image('logo', './assets/images/logo.png');
    }

    async create() {
        this.margin = 0;
        this.state = null;
        this.human = 'first';
        this.busy = true;
        this.selected = null;
        this.isModalOpen = false;
        this.pieces = new Map();
        this.handSprites = { first: [], second: [] };
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
        this.createLogo();
        this.createGameOverModal();
        this.input.on('pointerdown', (pointer) => {
            this.handleBoardPointer(pointer);
        });

        this.engine = new EngineClient();
        try {
            const initialized = await this.engine.initialize(DEFAULT_SIMULATIONS);
            this.state = initialized.state;
            this.human = initialized.state.human;
            this.busy = false;
            this.drawAll();
            this.finishLoading(initialized.backend, initialized.generation);
        } catch (error) {
            this.showFatalError(error);
        }
    }

    toPixel(column, row) {
        return {
            x: this.margin + LAYOUT.boardX * this.scaleX
                + column * LAYOUT.squareWidth * this.scaleX
                + LAYOUT.squareWidth * this.scaleX / 2,
            y: this.margin + LAYOUT.boardY * this.scaleY
                + row * LAYOUT.squareHeight * this.scaleY
                + LAYOUT.squareHeight * this.scaleY / 2
        };
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
        const column = Math.floor(localX / (LAYOUT.squareWidth * this.scaleX));
        const row = Math.floor(localY / (LAYOUT.squareHeight * this.scaleY));
        if (column >= LAYOUT.columns || row >= LAYOUT.rows) {
            return null;
        }
        return { column, row, index: row * LAYOUT.columns + column };
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
                this.restartGame();
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

    startThinking() {
        this.busy = true;
        this.stopPulseAnimations();
        this.clearHighlights();
        this.selected = null;
        if (this.logoSpin.isPaused()) {
            this.logoSpin.resume();
        }
        this.logo.disableInteractive();
    }

    stopThinking() {
        this.busy = false;
        this.logoSpin.pause();
        this.logo.setAngle(0);
        this.logo.setInteractive({ useHandCursor: true });
    }

    makeBoardPiece(index, piece) {
        const row = Math.floor(index / LAYOUT.columns);
        const column = index % LAYOUT.columns;
        const position = this.toPixel(column, row);
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
        return sprite;
    }

    drawAll() {
        this.stopPulseAnimations();
        for (const sprite of this.pieces.values()) {
            sprite.destroy();
        }
        this.pieces.clear();
        for (const player of ['first', 'second']) {
            for (const entry of this.handSprites[player]) {
                entry.sprite.destroy();
            }
            this.handSprites[player] = [];
        }

        this.state.board.forEach((piece, index) => {
            if (!piece) {
                return;
            }
            const sprite = this.makeBoardPiece(index, piece);
            const canSelect = this.isHumanTurn()
                && piece.owner === this.human
                && this.moveActionsFrom(index).length > 0;
            if (canSelect) {
                sprite.setInteractive({ useHandCursor: true });
            }
            this.pieces.set(index, sprite);
        });

        for (const player of ['first', 'second']) {
            const entries = this.handEntries(this.state, player);
            entries.forEach((kind, index) => {
                const sprite = this.makeHandPiece(player, index, kind);
                const canDrop = this.isHumanTurn()
                    && player === this.human
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
                this.handSprites[player].push({ kind, sprite });
            });
        }
        this.startPulseAnimations();
    }

    handEntries(state, player) {
        const playerIndex = player === 'first' ? 0 : 1;
        const entries = [];
        HAND_KINDS.forEach((kind, kindIndex) => {
            const count = state.hands[playerIndex][kindIndex];
            for (let index = 0; index < count; index += 1) {
                entries.push(kind);
            }
        });
        return entries;
    }

    isHumanTurn() {
        return Boolean(
            this.state
            && !this.busy
            && !this.isModalOpen
            && this.state.outcome.status === 'ongoing'
            && this.state.side_to_move === this.human
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
        const square = this.fromPixel(pointer.worldX, pointer.worldY);
        if (!square) {
            this.clearSelection();
            return;
        }

        if (this.selected) {
            const action = this.selected.actions.find((candidate) => {
                return candidate.to === square.index;
            });
            if (action) {
                this.playHumanAction(action);
                return;
            }
        }

        const piece = this.state.board[square.index];
        if (piece && piece.owner === this.human) {
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
        for (const entry of this.handSprites[this.human]) {
            if (entry.kind === kind) {
                entry.sprite.setTint(0xffe066);
            }
        }
        this.showActionTargets(actions, 0xffc107);
    }

    showActionTargets(actions, color) {
        for (const action of actions) {
            const row = Math.floor(action.to / LAYOUT.columns);
            const column = action.to % LAYOUT.columns;
            const center = this.toPixel(column, row);
            const highlight = this.add.rectangle(
                center.x,
                center.y,
                LAYOUT.squareWidth * this.scaleX * 0.9,
                LAYOUT.squareHeight * this.scaleY * 0.9,
                color,
                0.25
            ).setStrokeStyle(2, color, 0.9);
            this.highlights.push(highlight);
        }
    }

    clearSelection() {
        this.selected = null;
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

    startPulseAnimations() {
        if (!this.isHumanTurn() || this.selected) {
            return;
        }
        this.state.board.forEach((piece, index) => {
            if (piece && piece.owner === this.human && this.moveActionsFrom(index).length > 0) {
                const sprite = this.pieces.get(index);
                if (sprite) {
                    this.addPulseAnimation(sprite);
                }
            }
        });
        for (const entry of this.handSprites[this.human]) {
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
        this.startThinking();
        try {
            const response = await this.engine.playHuman(action);
            await this.animateAppliedAction(previousState, response);
            this.state = response.state;
            this.drawAll();
            if (this.showOutcomeIfFinished()) {
                this.stopThinking();
                return;
            }
            await this.playAiTurn();
        } catch (error) {
            this.stopThinking();
            this.showFatalError(error);
        }
    }

    async playAiTurn() {
        const previousState = this.state;
        const started = performance.now();
        this.startThinking();
        const response = await this.engine.playAi();
        const elapsed = performance.now() - started;
        if (elapsed < MINIMUM_THINK_MS) {
            await sleep(MINIMUM_THINK_MS - elapsed);
        }
        await this.animateAppliedAction(previousState, response);
        this.state = response.state;
        this.stopThinking();
        this.drawAll();
        this.showOutcomeIfFinished();
    }

    async animateAppliedAction(previousState, response) {
        this.stopPulseAnimations();
        const action = response.action;
        const player = previousState.side_to_move;
        const destinationRow = Math.floor(action.to / LAYOUT.columns);
        const destinationColumn = action.to % LAYOUT.columns;
        const destination = this.toPixel(destinationColumn, destinationRow);
        const capturedSprite = this.pieces.get(action.to) || null;
        let movingSprite;
        let movingKind;

        if (action.type === 'move') {
            movingSprite = this.pieces.get(action.from);
            movingKind = previousState.board[action.from].kind;
        } else {
            const handEntry = this.handSprites[player].find((entry) => {
                return entry.kind === action.piece;
            });
            movingSprite = handEntry ? handEntry.sprite : null;
            movingKind = action.piece;
        }

        if (!movingSprite) {
            const source = action.type === 'move'
                ? this.toPixel(action.from % LAYOUT.columns, Math.floor(action.from / LAYOUT.columns))
                : this.handCenter(player, 0);
            movingSprite = this.add.sprite(
                source.x,
                source.y,
                'characters',
                PIECE_FRAME[movingKind]
            );
            if (player === 'second') {
                movingSprite.setAngle(180);
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
            const capturedKind = response.captured === 'kodama_samurai'
                ? 'kodama'
                : response.captured;
            const destinationEntries = this.handEntries(response.state, player);
            const handIndex = destinationEntries.lastIndexOf(capturedKind);
            capturedSprite.setDepth(55);
            if (handIndex >= 0) {
                const handDestination = this.handCenter(player, handIndex);
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
            movingSprite.setFrame(PIECE_FRAME.kodama_samurai);
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
        if (outcome.status === 'draw') {
            this.showGameOver('Draw', 'The position was repeated three times.');
        } else if (outcome.player === this.human) {
            this.showGameOver('Victory!', outcomeReason(outcome.reason));
        } else {
            this.showGameOver('Defeat', outcomeReason(outcome.reason));
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
            fontFamily: 'system-ui, sans-serif',
            fontSize: '36px',
            color: '#111111',
            fontStyle: 'bold'
        }).setOrigin(0.5);
        this.modalSubtitle = this.add.text(centerX, centerY - 8, '', {
            fontFamily: 'system-ui, sans-serif',
            fontSize: '16px',
            color: '#444444',
            align: 'center',
            wordWrap: { width: panelWidth - 42 }
        }).setOrigin(0.5);
        const buttonWidth = 180;
        const buttonHeight = 48;
        const buttonY = centerY + 65;
        const button = this.add.graphics();
        button.fillStyle(0x222222, 1);
        button.fillRoundedRect(
            centerX - buttonWidth / 2,
            buttonY - buttonHeight / 2,
            buttonWidth,
            buttonHeight,
            12
        );
        const hitArea = this.add.rectangle(
            centerX,
            buttonY,
            buttonWidth,
            buttonHeight,
            0x000000,
            0
        ).setInteractive({ useHandCursor: true });
        const label = this.add.text(centerX, buttonY, 'New game', {
            fontFamily: 'system-ui, sans-serif',
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
            this.restartGame();
        });
        this.modal.add([
            blocker,
            panel,
            this.modalTitle,
            this.modalSubtitle,
            button,
            hitArea,
            label
        ]);
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
        this.startThinking();
        try {
            this.state = await this.engine.reset();
            this.stopThinking();
            this.drawAll();
        } catch (error) {
            this.stopThinking();
            this.showFatalError(error);
        }
    }

    finishLoading(backend, generation) {
        const title = document.getElementById('loading-title');
        const detail = document.getElementById('loading-detail');
        const loading = document.getElementById('loading');
        title.textContent = 'Ready!';
        detail.textContent = `Champion generation ${generation} · ${backend}`;
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

function outcomeReason(reason) {
    if (reason === 'koropokkuru_captured') {
        return 'The Koropokkuru was captured.';
    }
    if (reason === 'koropokkuru_reached_goal') {
        return 'The Koropokkuru reached the opposite camp.';
    }
    if (reason === 'opponent_has_no_legal_action') {
        return 'The opponent has no legal action left.';
    }
    return 'The game is over.';
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

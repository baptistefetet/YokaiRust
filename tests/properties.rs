//! Randomized invariants that complement hand-written rules examples.

use proptest::prelude::*;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use yokai::{Action, Game, POLICY_ACTIONS, Player, PolicyIndex};

const POLICY_ACTIONS_U8: u8 = 132;

proptest! {
    #[test]
    fn every_generated_action_is_applicable_and_policy_round_trips(seed in any::<u64>()) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let starting_player = if rng.random::<bool>() {
            Player::First
        } else {
            Player::Second
        };
        let mut game = Game::new(starting_player);

        for _ in 0..128 {
            let legal_actions = game.legal_actions();
            if legal_actions.is_empty() {
                break;
            }
            let side_to_move = game.position().side_to_move();
            let mut generated_by_engine = legal_actions.clone();
            generated_by_engine.sort_unstable_by_key(|action| action.policy_index(side_to_move));
            let mut scanned_policy_space = (0_u8..POLICY_ACTIONS_U8)
                .filter_map(PolicyIndex::new)
                .filter_map(|index| Action::from_policy_index(index, side_to_move))
                .filter(|&action| game.is_legal_action(action))
                .collect::<Vec<_>>();
            scanned_policy_space.sort_unstable_by_key(|action| action.policy_index(side_to_move));
            prop_assert_eq!(generated_by_engine, scanned_policy_space);

            for &action in &legal_actions {
                prop_assert!(game.is_legal_action(action));
                let policy = action.policy_index(side_to_move).expect("legal moves have a policy index");
                prop_assert_eq!(Action::from_policy_index(policy, side_to_move), Some(action));
                let mirrored = action.mirrored_horizontally();
                prop_assert_eq!(mirrored.mirrored_horizontally(), action);
            }

            let selected = rng.random_range(0..legal_actions.len());
            let count_before = game.position().physical_piece_count();
            let transition = game.apply(legal_actions[selected]).expect("generated move applies");
            let count_after = game.position().physical_piece_count();
            if transition.captured == Some(yokai::PieceKind::Koropokkuru) {
                prop_assert_eq!(count_after + 1, count_before);
            } else {
                prop_assert_eq!(count_after, count_before);
            }
            if transition.outcome.is_terminal() {
                break;
            }
        }
    }

    #[test]
    fn policy_decoder_never_panics(index in 0_u8..POLICY_ACTIONS_U8, first in any::<bool>()) {
        let player = if first { Player::First } else { Player::Second };
        let policy = PolicyIndex::new(index).expect("generated policy index is valid");
        if let Some(action) = Action::from_policy_index(policy, player) {
            prop_assert_eq!(action.policy_index(player), Some(policy));
        }
    }
}

#[test]
fn optimized_generator_matches_independent_policy_space_scan() {
    assert_eq!(usize::from(POLICY_ACTIONS_U8), POLICY_ACTIONS);
    for player in [Player::First, Player::Second] {
        let game = Game::new(player);
        let mut generated = game.legal_actions();
        generated.sort_unstable_by_key(|action| action.policy_index(player));

        let mut scanned = (0_u8..POLICY_ACTIONS_U8)
            .filter_map(PolicyIndex::new)
            .filter_map(|index| Action::from_policy_index(index, player))
            .filter(|&action| game.is_legal_action(action))
            .collect::<Vec<_>>();
        scanned.sort_unstable_by_key(|action| action.policy_index(player));

        assert_eq!(generated, scanned);
    }
}

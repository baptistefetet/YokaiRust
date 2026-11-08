//! Randomized invariants that complement hand-written rules examples.

use proptest::prelude::*;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use yokai::{Action, Game, POLICY_ACTIONS, Player, PolicyIndex, Position};

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
        let policy_actions = u8::try_from(POLICY_ACTIONS).unwrap();

        for _ in 0..128 {
            let legal_actions = game.legal_actions();
            if legal_actions.is_empty() {
                break;
            }
            let side_to_move = game.position().side_to_move();
            let mut generated_by_engine = legal_actions.clone();
            generated_by_engine.sort_unstable_by_key(|action| action.policy_index(side_to_move));
            let mut scanned_policy_space = (0_u8..policy_actions)
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

                let mut next = game.clone();
                let transition = next.apply(action).expect("generated action applies");
                let expected_occurrences = if matches!(transition.outcome, yokai::Outcome::Win { .. }) {
                    0
                } else {
                    1 + game.position_history().iter().filter(|position| *position == next.position()).count() as u8
                };
                prop_assert_eq!(game.repetition_count_after(action), Some(expected_occurrences));
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
            let position = *game.position();
            prop_assert_eq!(
                Position::from_parts(*position.board(), *position.hands(), position.side_to_move()),
                Ok(position)
            );
            let json = serde_json::to_string(&position).expect("position serializes");
            prop_assert_eq!(
                serde_json::from_str::<Position>(&json).expect("legal play preserves valid material"),
                position
            );
            if transition.outcome.is_terminal() {
                break;
            }
        }
    }
}

#[test]
fn every_decodable_policy_index_round_trips_for_both_players() {
    let policy_actions = u8::try_from(POLICY_ACTIONS).unwrap();
    for player in [Player::First, Player::Second] {
        for index in 0..policy_actions {
            let policy = PolicyIndex::new(index).unwrap();
            if let Some(action) = Action::from_policy_index(policy, player) {
                assert_eq!(action.policy_index(player), Some(policy));
            }
        }
    }
}

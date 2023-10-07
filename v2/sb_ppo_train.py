from pathlib import Path
from utils import *

def main():
    MODEL = "PPO"
    FRAMEWORK = "sb"
    LEARN_DESCRIBE = "best_param_early_stopping"
    SEED = 1337
    NUM_ENVS = 256
    N_STEPS = 64

    tb_log_name = (
        f"{MODEL}_{FRAMEWORK}_" f"{LEARN_DESCRIBE}_{NUM_ENVS}_" f"{N_STEPS}_{SEED}"
    )

    multiplier = 200
    total_timesteps = NUM_ENVS * N_STEPS * multiplier
    train_envs, trade_env = get_train_trade_environment(
        framework="sb", num_envs=NUM_ENVS, seed=SEED
    )


    trained_model = get_best_ppo_model(train_envs, SEED)
    model_path = Path(TRAINED_MODEL_DIR) / f"{MODEL}.zip"

    try:
        trained_model.learn(
            total_timesteps=total_timesteps,
            callback=TensorboardCallback(eval_env=trade_env, model_path=model_path, seed=SEED),
            tb_log_name=tb_log_name,
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=True,
        )

        t_info = test_model(trade_env, trained_model, SEED)
        print(json.dumps(t_info, indent=4, default=float))
        trained_model.save(model_path)
    except KeyboardInterrupt:
        t_info = test_model(trade_env, trained_model, SEED)
        print(json.dumps(t_info, indent=4, default=float))
        trained_model.save(model_path)
    
    train_envs.close()
    trade_env.close()

    # trained_model = get_best_ppo_model(TRAIN_ENVS, SEED)
    # # model = get_a2c_model(train_envs, N_STEPS, SEED)
    # # model = load_ppo_model(train_envs)

    # NUM_ENVS = 256
    # N_STEPS = 512
    # TIME_STAMPS = 50

    # TOTAL_TIME_STAMPS = TIME_STAMPS * NUM_ENVS * N_STEPS

    # trained_model.learn(
    #     total_timesteps=TOTAL_TIME_STAMPS,
    #     callback=TensorboardCallback(
    #         save_freq=1, model_prefix=MODEL_PREFIX, eval_env=TRADE_ENV, seed=SEED
    #     ),
    #     tb_log_name=f"sb_single_step_reward_early_stopping_best_{MODEL}_model",
    #     log_interval=1,
    #     progress_bar=True,
    #     reset_num_timesteps=False,
    # )

    # print(test_model(TRADE_ENV, trained_model, SEED))
    # trained_model.save(Path(TRAINED_MODEL_DIR) / f"{MODEL_PREFIX}.zip")
    # trained_model.save_replay_buffer(
    #     Path(TRAINED_MODEL_DIR) / f"{MODEL_PREFIX}_replay_buffer"
    # )
    # trade_model = load_ppo_model()
    # print(test_model(TRADE_ENV, trade_model, SEED))

    # TRAIN_ENVS.close()
    # TRADE_ENV.close()


if __name__ == "__main__":
    main()

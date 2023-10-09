from utils import *
from stable_baselines3 import PPO


TRAIN_ENVS, TRADE_ENV = get_train_trade_environment()


def main():
    SEED = 1337
    NUM_ENVS = 256
    MODEL = "PPO"
    FRAMEWORK = "sb"
    LEARN_DESCRIBE = "best_param_early_stopping"
    
    model_filename = Path(TRAINED_MODEL_DIR) / f"{MODEL}.zip"

    train_envs, trade_env = get_train_trade_environment(
        framework="sb", num_envs=NUM_ENVS, seed=SEED
    )

    trained_model = PPO.load(model_filename, env=train_envs, tensorboard_log=TENSORBOARD_LOG_DIR)

    # info = test_model(trade_env, trained_model, SEED)

    # Resume Training
    multiplier = 10 
    total_timesteps = NUM_ENVS * trained_model.n_steps * multiplier
    tb_log_name = (
        f"{MODEL}_{FRAMEWORK}_" 
        f"{LEARN_DESCRIBE}_{NUM_ENVS}_" 
        f"{trained_model.n_steps}_{SEED}"
    )

    trained_model.set_env(train_envs)
    try:
        trained_model.learn(
            total_timesteps=total_timesteps,
            callback=TensorboardCallback(eval_env=trade_env, model_filename=model_filename, seed=SEED),
            tb_log_name=tb_log_name,
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        trained_model.save(model_filename)

    trained_model.save(model_filename)


if __name__ == "__main__":
    main()

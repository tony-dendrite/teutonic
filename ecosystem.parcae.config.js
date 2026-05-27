const { execSync } = require("child_process");

function doppler(key) {
  return execSync(`doppler secrets get ${key} --plain -p arbos -c dev`, { encoding: "utf8" }).trim();
}

module.exports = {
  apps: [{
    name: "teutonic-eval-tunnel",
    script: "./tunnel.sh",
    cwd: "/home/const/workspace",
    autorestart: true,
    restart_delay: 5000,
    max_restarts: 1000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }, {
    name: "teutonic-validator",
    script: "validator.py",
    args: "",
    interpreter: "/home/const/workspace/.venv/bin/python",
    cwd: "/home/const/workspace",
    env: {
      TEUTONIC_EVAL_SERVER: "http://localhost:9000",
      // Parcae staging template: keep the live chain on chain.toml until the
      // exported genesis repo is uploaded and chain.parcae.toml has a real digest.
      TEUTONIC_CHAIN_OVERRIDE: "chain.parcae.toml",
      TEUTONIC_EVAL_DATASET_MODE: "raw_hippius",
      TEUTONIC_RAW_DATASET_PREFIX: "hf-mirrors/HuggingFaceFW/fineweb-edu/data",
      TEUTONIC_RAW_DATASET_MANIFEST: "hf-mirrors/HuggingFaceFW/fineweb-edu/data/_manifest.json",
      TEUTONIC_RAW_TOKENIZER_REPO: "SandyResearch/parcae-tokenizer",
      TEUTONIC_EVAL_N: "10000",
      TEUTONIC_EVAL_N_PUBLIC: "10000",
      TEUTONIC_EVAL_N_PRIVATE: "0",
      TEUTONIC_NETUID: "3",
      TEUTONIC_NETWORK: "finney",
      BT_WALLET_NAME: "teutonic",
      BT_WALLET_HOTKEY: "default",
      TEUTONIC_R2_ENDPOINT: doppler("R2_URL"),
      TEUTONIC_R2_BUCKET: doppler("R2_BUCKET_NAME"),
      TEUTONIC_R2_ACCESS_KEY: doppler("R2_ACCESS_KEY_ID"),
      TEUTONIC_R2_SECRET_KEY: doppler("R2_SECRET_ACCESS_KEY"),
      TEUTONIC_HIPPIUS_ACCESS_KEY: doppler("HIPPIUS_ACCESS_KEY"),
      TEUTONIC_HIPPIUS_SECRET_KEY: doppler("HIPPIUS_SECRET_KEY"),
      TEUTONIC_DS_ENDPOINT: "https://s3.hippius.com",
      TEUTONIC_DS_BUCKET: "teutonic-sn3",
      TEUTONIC_DS_ACCESS_KEY: doppler("HIPPIUS_ACCESS_KEY"),
      TEUTONIC_DS_SECRET_KEY: doppler("HIPPIUS_SECRET_KEY"),
      TMC_API_KEY: doppler("TMC_API_KEY"),
      DISCORD_BOT_TOKEN: doppler("DISCORD_BOT_TOKEN"),
      DISCORD_CHANNEL_ID: doppler("DISCORD_CHANNEL_ID"),
      TEUTONIC_TICK_RESTART_AFTER: "1800",
      TEUTONIC_MAX_CONSECUTIVE_TICK_ERRORS: "20",
      TEUTONIC_STREAM_IDLE_WARN_AFTER: "600",
      TEUTONIC_STREAM_IDLE_TIMEOUT: "1800",
      TEUTONIC_KING_HASH_TIMEOUT_S: "1200",
    },
    max_restarts: 10,
    restart_delay: 5000,
    autorestart: true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }],
};

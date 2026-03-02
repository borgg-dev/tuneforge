module.exports = {
  apps: [
    {
      name: "tuneforge-miner",
      script: "python3",
      args: "-m neurons.miner --env-file .env.miner",
      cwd: __dirname,
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    {
      name: "tuneforge-validator",
      script: "python3",
      args: "-m neurons.validator --env-file .env.validator",
      cwd: __dirname,
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    {
      name: "tuneforge-api",
      script: "python3",
      args: "-m uvicorn tuneforge.api.app:app --host 0.0.0.0 --port 8000",
      cwd: __dirname,
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
  ],
};

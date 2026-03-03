module.exports = {
  apps: [
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
        CUDA_VISIBLE_DEVICES: "-1",
      },
    },
    {
      name: "tuneforge-miner-1",
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
      name: "tuneforge-miner-2",
      script: "python3",
      args: "-m neurons.miner --env-file .env.miner2",
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

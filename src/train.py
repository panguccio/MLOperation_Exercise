import neptune

run = neptune.init_run(
    project="emmadariol/MLops",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZmMwMmNjNC03NmY4LTQ3ZDEtOTMxNS1kYWU0M2U4ZWU4MTkifQ==",
)  # your credentials

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].append(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()


#setx NEPTUNE_API_TOKEN "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZmMwMmNjNC03NmY4LTQ3ZDEtOTMxNS1kYWU0M2U4ZWU4MTkifQ=="
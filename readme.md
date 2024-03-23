This is a Lean-CLI project. To run it you will need to install Lean-CLI: https://github.com/QuantConnect/lean-cli

To run the project, clone it into your Lean-CLI directory (the directory you ran `lean init` in, the one with "data" and "storage" folders) as a sibling to any other Lean CLI projects. You can then run Lean CLI commands from that directory as usual.

Bring your IDE terminal up to that level:
```shell
cd ../
```
then you can run the backtest:
```shell
lean backtest "AlgorithmFrameworkDemo"
```

As a reminder,
```shell
lean report --overwrite --pdf
```
will generate PDF report of the strategy's performance.

This dataset has 10,000 records from a factory machine, tracking information on each run to see if the machine fails. Each entry has details like temperatures, speed, torque (a measure of force), and how much the tool has worn out. Here’s a simple look at each column:

1. **UID**: Unique ID number for each record.
2. **Product ID**: Code showing product quality (Low, Medium, High) and a unique number.
3. **Type**: Product quality type—L (Low), M (Medium), H (High).
4. **Air Temperature**: The air temperature around the machine (in Kelvin).
5. **Process Temperature**: Temperature during the machine’s process, a bit higher than air temperature.
6. **Rotational Speed**: How fast the machine spins (in RPM).
7. **Torque**: The force the machine applies (in Nm).
8. **Tool Wear**: How long the tool has been used (in minutes).
9. **Machine Failure**: Shows if the machine failed during the run (1 means it failed, 0 means it didn’t).

There are also five types of failures, which can make the machine stop working:
10. **TWF (Tool Wear Failure)**: Fails when the tool is too worn out.
11. **HDF (Heat Dissipation Failure)**: Fails if temperatures aren’t managed well.
12. **PWF (Power Failure)**: Fails if power is too high or low.
13. **OSF (Overstrain Failure)**: Fails if the machine’s work puts too much strain on the tool.
14. **RNF (Random Failure)**: Fails by chance, even if everything seems okay.

This data helps predict when and why the machine might fail so problems can be fixed faster.

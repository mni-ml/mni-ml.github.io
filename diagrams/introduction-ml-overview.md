# Introduction — What is ML (Overview Diagram)

Place this in Notion as a code block with language set to `mermaid`.
Insert after the first paragraph ("...correct outputs that we did not previously have.").

```mermaid
graph TD
    subgraph training ["Training"]
        Data["Input-Output Pairs"] --> Model["Model"]
        Model --> Predicted["Predicted Output"]
        Expected["Expected Output"] --> Compare["Compare"]
        Predicted --> Compare
        Compare -->|"adjust weights"| Model
    end

    Model -.->|"calibrated"| Trained["Trained Model"]
    NewInput["New Input"] --> Trained
    Trained --> NewOutput["Correct Output"]
```

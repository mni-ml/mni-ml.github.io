# Introduction — What is ML (Overview Diagram)

Place this in Notion as a code block with language set to `mermaid`.
Insert after the first paragraph ("...correct outputs that we did not previously have.").

```mermaid
graph LR
    subgraph training ["Training (calibrating the function)"]
        direction TB
        Data["Input-Output Pairs"] --> Model["Mathematical Function<br/>(with adjustable parameters)"]
        Model --> Predicted["Predicted Output"]
        Predicted --> Compare["Compare to<br/>Expected Output"]
        Compare -->|"Adjust parameters"| Model
    end

    training --> Trained["Trained Model"]
    NewInput["New Input"] --> Trained
    Trained --> NewOutput["Correct Output<br/>(never seen before)"]
```

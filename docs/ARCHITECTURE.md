# Technical Architecture Documentation 🏗️

This document provides a detailed breakdown of the technical design, execution loop, type reflection engine, and context injection pattern powering the AI Agent framework.

---

## 1. High-Level Architecture

The framework is structured as a lightweight, framework-free autonomous agent system. It delegates language reasoning to an OpenAI-compatible endpoint (such as local Ollama instances) and provides deterministic tool execution and long-term state management.

```
+-------------------------------------------------------------+
|                        User Prompt                          |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     Agent Orchestrator                      |
|  - Message History Management                               |
|  - Multi-iteration Tool Call Resolution                     |
|  - System Prompt & Instruction Enforcement                  |
+-------------------------------------------------------------+
          |                                       |
          v                                       v
+-----------------------+              +----------------------+
|  LLM Inference Client |              |    Tool Registry     |
| (Ollama / OpenAI API) |              |  - Schema Reflection |
+-----------------------+              |  - Context Injection |
                                       |  - Execution Sandbox |
                                       +----------------------+
                                                  |
                                                  v
                                       +----------------------+
                                       |   Memory Subsystem   |
                                       |  - Dense Embeddings  |
                                       |  - Cosine Search     |
                                       |  - JSON Persistence  |
                                       +----------------------+
```

---

## 2. Core Subsystems

### 2.1 Agent Orchestration Loop (`Agent`)

The `Agent` class coordinates the multi-turn exchange between the user prompt, the LLM reasoning engine, and tool invocations.

#### Key Mechanics:
1. **Tool Schema Registration**:
   When initialized with a list of `Tool` objects, the agent constructs:
   - `TOOL_MAP`: A hash map of `tool_name -> Tool` for O(1) runtime dispatch.
   - `Tool_SCHEMA`: A list of OpenAI-formatted function calling JSON schemas.
2. **Context Binding**:
   Initializes internal state and registers context dependencies (e.g. `{"memory": self.memory}`).
3. **Execution Loop (`run`)**:
   - Appends user input to `self.messages`.
   - Loops up to `MAX_ITERATIONS` (default: 10):
     - Calls the LLM via `client.chat.completions.create(..., tools=self.Tool_SCHEMA)`.
     - Checks if the response contains `tool_calls`.
     - If no `tool_calls` exist, the final textual response is returned.
     - If `tool_calls` are present:
       - Iterates through each `tool_call`.
       - Dispatches via `self.execute(tool_call)`.
       - Appends a message of role `"tool"` containing the JSON-serialized result along with matching `tool_call_id`.
     - Repeats the cycle so the LLM can synthesize results or chain additional tools.

---

### 2.2 Dynamic Type Reflection (`generate_parameters`)

Unlike systems that require developers to write repetitive JSON Schemas by hand, this framework uses Python's runtime introspection (`inspect` module) and type hints (`typing` module) to dynamically synthesize JSON Schema representations.

#### Type Mapping Logic (`python_type_to_json_type`):

| Python Type | JSON Schema Type | Notes |
| :--- | :--- | :--- |
| `str` | `string` | Standard string parameter |
| `int` | `integer` | Integer number |
| `float` | `number` | Floating-point number |
| `bool` | `boolean` | Boolean flag |
| `Literal["a", "b", ...]` | `string` / `number` + `enum` | Dynamically extracts allowed enum choices via `typing.get_args()` |

#### Parameter Extraction Workflow:
```python
def generate_parameters(function):
    signature = inspect.signature(function)
    properties = {}
    required = []

    for name, parameter in signature.parameters.items():
        if name == "memory":  # Internal context, exclude from LLM schema
            continue

        json_type = python_type_to_json_type(parameter.annotation)
        properties[name] = json_type if isinstance(json_type, dict) else {"type": json_type}

        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }
```

---

### 2.3 Runtime Context & Dependency Injection

Tools often require access to shared services (such as databases, memory stores, or active API sessions) that should **not** be supplied by the LLM.

The `Tool.execute()` method inspects the signature of the target function and transparently merges runtime context:

```python
def execute(self, arguments, context):
    if context is None:
        context = {}
    
    sig = inspect.signature(self.function)
    kwargs = dict(arguments)
    
    # Inject matching context objects (e.g. 'memory')
    for k, v in context.items():
        if k in sig.parameters:
            kwargs[k] = v
            
    return self.function(**kwargs)
```

**Benefits:**
- **Zero Hallucination of System Objects**: The LLM never sees or hallucinates memory pointers or session handles.
- **Clean Tool Definitions**: Tools declare `def save_memory(memory: Memory, key: str, value: str): ...` cleanly and receive `memory` automatically.

---

## 3. Error Handling & Safety

- **JSON Argument Decoding**: Tool execution safely catches malformed JSON strings from LLM outputs.
- **Runtime Execution Traps**: All tool exceptions are intercepted inside `Tool.execute()`, returning a structured error message (`"Tool execution failed: <error>"`) back into conversation context rather than crashing the loop.
- **Iteration Cap**: Prevents infinite tool-calling loops with an upper bound limit (`MAX_ITERATIONS = 10`).

"""第3回 宿題（発展a）: ADK 版エージェントを CLI で1回動かし、app.agent --trace と同じ形式で表示する。

使い方:
    PYTHONPATH=. uv run --with google-adk python -m scripts.adk_trace "LoRAとQLoRAの違いは？"

宿題では必須ではない（adk run で足りる）。app.agent --trace と並べて比べやすくするために作った。
"""

import argparse
import asyncio

from google.adk.runners import InMemoryRunner
from google.genai import types

from adk_agent.glossary_agent.agent import root_agent


async def run(question: str) -> None:
    runner = InMemoryRunner(agent=root_agent, app_name="aie2026")
    session = await runner.session_service.create_session(app_name="aie2026", user_id="me")
    message = types.Content(role="user", parts=[types.Part(text=question)])

    trace, answer = [], []
    requests = input_tokens = output_tokens = 0
    async for event in runner.run_async(user_id="me", session_id=session.id, new_message=message):
        # モデルの応答1回 = LLM 呼び出し1回。usage_metadata はモデル応答のイベントにだけ付く
        if event.usage_metadata:
            requests += 1
            input_tokens += event.usage_metadata.prompt_token_count or 0
            output_tokens += event.usage_metadata.candidates_token_count or 0
        for part in (event.content.parts if event.content else None) or []:
            if part.function_call:
                trace.append(f"[tool call]   {part.function_call.name}({dict(part.function_call.args or {})})")
            elif part.function_response:
                body = str(part.function_response.response)[:120]
                trace.append(f"[tool return] {part.function_response.name} -> {body}…")
            elif part.text and not part.thought and event.is_final_response():
                answer.append(part.text)

    print("".join(answer))
    print("\n--- trace ---")
    print("\n".join(trace))
    print(f"[usage] requests={requests}, input_tokens={input_tokens}, output_tokens={output_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ADK 版の学習支援エージェントに質問する")
    parser.add_argument("question")
    args = parser.parse_args()
    print(f"[model] {root_agent.model}  (ADK)")
    asyncio.run(run(args.question))


if __name__ == "__main__":
    main()

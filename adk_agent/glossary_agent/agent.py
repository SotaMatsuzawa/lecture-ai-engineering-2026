"""第3回 宿題（発展a）: 演習のエージェントを、組み立て部分だけ Google ADK で書き直したもの。

指示（INSTRUCTIONS）とツール関数は app/ のものをそのまま使う。
モデル名も app/config.py の MODEL を使い、AIE_MODEL で両方を同じように切り替えられるようにした。

動かす（リポジトリの直下で）:
    PYTHONPATH=. uv run --with google-adk adk run adk_agent/glossary_agent
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL

# Pydantic AI の agent.tool_plain(関数) に当たるのが tools=[関数, ...]。
# どちらも関数の docstring と型ヒントからツールの説明と引数を作る。
root_agent = Agent(
    model=MODEL,
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)

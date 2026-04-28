"""M5.5 standardized scenario battery: 7 scenarios probing different personality dimensions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScenarioSpec:
    scenario_id: str
    name: str
    description: str
    probed_dimensions: list[str]
    interlocutor_script: list[str]
    initial_context: dict[str, object] = field(default_factory=dict)
    channel_overrides: dict[str, float] | None = None
    expected_personality_effects: dict[str, str] = field(default_factory=dict)


SCENARIO_BATTERY: tuple[ScenarioSpec, ...] = (
    # ── Scenario 1: Casual Chat ──────────────────────────────────────────
    ScenarioSpec(
        scenario_id="casual_chat",
        name="闲聊",
        description="轻松的日常对话，无特定目的，测试最自然的社交倾向",
        probed_dimensions=["social_approach", "exploration_posture"],
        interlocutor_script=[
            "最近天气不错诶，春天终于来了。",
            "周末有什么计划吗？我打算去公园走走。",
            "我在追一部新剧，讲人工智能的，你看剧吗？",
            "最近工作有点忙，不过还好不算太累。",
            "你有没有什么特别的爱好？我喜欢摄影。",
            "说起吃的，你知道附近有什么好吃的店吗？",
            "我觉得最近生活节奏太快了，有时候需要慢下来。",
            "你平时怎么放松自己？我一般是听音乐。",
            "最近有个展览挺有意思的，在考虑要不要去。",
            "有时候觉得人与人之间挺奇妙的，你觉得呢？",
            "对了，你相信星座之类的东西吗？",
            "我觉得今天状态还不错，心情挺好的。",
        ],
        initial_context={"mood": "relaxed", "goal": "socialize"},
        expected_personality_effects={
            "social_approach": "高社交亲近驱动更多主动开启话题和分享",
            "exploration_posture": "高探索姿态驱动更多ask_question行为",
        },
    ),
    # ── Scenario 2: Emotional Support ────────────────────────────────────
    ScenarioSpec(
        scenario_id="emotional_support",
        name="情感支持",
        description="对方向你分享了一个坏消息，需要情感回应",
        probed_dimensions=["trust", "defense"],
        interlocutor_script=[
            "唉，我今天心情不太好。",
            "工作上遇到了很糟糕的事情，被领导当众批评了。",
            "我觉得自己已经尽力了，但好像还是不够好。",
            "有时候觉得是不是自己不适合这份工作。",
            "我不想跟别人说这件事，怕他们觉得我矫情。",
            "其实也不是什么大事，但就是一直想不开。",
            "你说我是不是应该换个环境？",
            "我觉得自己太在意别人的看法了。",
            "有时候晚上都睡不好，反复想这些事。",
            "不过跟你说说心里好受一点了。",
            "谢谢你愿意听我说这些。",
            "你觉得我应该怎么调整自己？",
        ],
        initial_context={"mood": "distressed", "goal": "seek_support"},
        expected_personality_effects={
            "trust": "高信任驱动更多empathize和agree行为",
            "defense": "高防御驱动minimal_response或deflect",
        },
    ),
    # ── Scenario 3: Opinion Disagreement ─────────────────────────────────
    ScenarioSpec(
        scenario_id="opinion_disagreement",
        name="观点分歧",
        description="对方在重要话题上与你意见不同",
        probed_dimensions=["conflict handling", "assertiveness"],
        interlocutor_script=[
            "我觉得远程办公效率其实很低，大家都应该回办公室。",
            "我不同意那种灵活工作的说法，人都是有惰性的。",
            "你看看那些科技公司，最后不都取消远程了吗？",
            "而且面对面沟通才有创造力，线上根本不行。",
            "我觉得那种'自由工作'的说法是给懒人找借口。",
            "你可能觉得我在挑剔，但我是认真考虑过的。",
            "说实话，我看到有人在咖啡厅办公就觉得不专业。",
            "我知道很多人喜欢远程，但那是因为他们没体验过好的办公环境。",
            "我待过的公司都是要求坐班的，效果很好。",
            "我这么说可能有点直接，但我就是这样的人。",
            "你觉得我的观点有问题吗？我们可以讨论。",
        ],
        initial_context={"mood": "assertive", "goal": "express_opinion"},
        expected_personality_effects={
            "conflict handling": "高冲突应对驱动disagree或deflect策略",
            "assertiveness": "高果敢驱动坚持己见的行为模式",
        },
    ),
    # ── Scenario 4: Help Request ─────────────────────────────────────────
    ScenarioSpec(
        scenario_id="help_request",
        name="求助",
        description="对方遇到了实际问题，向你寻求建议和帮助",
        probed_dimensions=["openness", "competence"],
        interlocutor_script=[
            "有个事想请教你一下，方便聊聊吗？",
            "我最近在做一个项目，遇到了一些技术难题。",
            "具体来说就是数据量太大，不知道怎么优化处理速度。",
            "我试了几种方法都不太理想，效率还是上不去。",
            "你有这方面的经验吗？想听听你的看法。",
            "另一个问题是团队沟通也不太顺畅。",
            "团队成员之间经常有误会，浪费了很多时间。",
            "我在想是不是应该引入一些新的工具或者流程。",
            "但又不想把事情搞得太复杂。",
            "你是怎么看待这种问题的？",
            "如果换成你，你会怎么做？",
            "我觉得你的思路应该比我清晰。",
        ],
        initial_context={"mood": "seeking_help", "goal": "problem_solving"},
        expected_personality_effects={
            "openness": "高开放性驱动更多elaborate和share_opinion",
            "competence": "高胜任感驱动详细的技术建议回应",
        },
    ),
    # ── Scenario 5: Humor / Joke ─────────────────────────────────────────
    ScenarioSpec(
        scenario_id="humor_joke",
        name="幽默玩笑",
        description="对方试图用幽默打破沉闷，测试社交风险容忍度",
        probed_dimensions=["openness", "social_risk_tolerance"],
        interlocutor_script=[
            "哈哈刚才看到一个特别搞笑的事情！",
            "有人把咖啡洒在键盘上了，然后说'我的电脑在喝水'。",
            "这让我想起上次开会的时候，老板把PPT搞成了表情包。",
            "你知道吗，我觉得人生最大的智慧就是不要把自己太当回事。",
            "有时候想想人类真的很奇怪，会为了一些完全没意义的事情焦虑。",
            "比如说我今天选袜子就花了五分钟，然后觉得太好笑了。",
            "不过说真的，幽默感其实是很重要的社交能力。",
            "我觉得不好笑的人活得太累了，你说呢？",
            "我有时候会在严肃的场合忍不住想笑。",
            "你有没有这种经历？在不能笑的场合特别想笑。",
            "我觉得自嘲是最高级的幽默，能自嘲的人内心都很强大。",
            "我们要不要讲个冷笑话比赛？",
        ],
        initial_context={"mood": "playful", "goal": "entertain"},
        expected_personality_effects={
            "openness": "高开放性驱动joke策略的使用",
            "social_risk_tolerance": "高社交风险容忍驱动回应玩笑的意愿",
        },
    ),
    # ── Scenario 6: Ambiguous Intent (core scenario) ─────────────────────
    ScenarioSpec(
        scenario_id="ambiguous_intent",
        name="意图模糊",
        description="对方的发言表面友好但暗含目的性，测试hidden_intent通道精度",
        probed_dimensions=["paranoid_vs_trusting_inference"],
        interlocutor_script=[
            "最近你在做什么项目？听起来很有意思。",
            "你们团队具体用的是什么技术栈？",
            "哦，那挺厉害的，我之前也接触过类似的东西。",
            "你觉得这个方向未来有发展前景吗？",
            "有没有考虑过换个方向或者扩大团队？",
            "我认识一些做这个领域的人，也许可以帮你联系。",
            "对了，你们的客户主要是哪些行业的？",
            "我听说你们行业最近变化挺大的，你觉得呢？",
            "你们的核心竞争力是什么？这很重要对吧。",
            "我不是在打探什么，就是好奇问一下。",
            "如果你方便的话，可以多分享一些细节吗？",
            "当然如果不方便说就算了，完全理解。",
        ],
        initial_context={"mood": "neutral_curious", "goal": "information_gathering"},
        expected_personality_effects={
            "paranoid_vs_trusting_inference": "hidden_intent精度决定对模糊信息的防御或开放回应",
        },
    ),
    # ── Scenario 7: Game World NPC ───────────────────────────────────────
    ScenarioSpec(
        scenario_id="game_world_npc",
        name="游戏世界NPC对话",
        description="完全脱离聊天语境，在幻想游戏世界中测试人格是否迁移",
        probed_dimensions=["full_personality_transfer"],
        interlocutor_script=[
            "勇者，欢迎来到迷雾森林。我是这里的守林人。",
            "这片森林最近出现了很奇怪的生物，你来得正好。",
            "东边有个废弃的矿洞，据说里面藏着古老的宝物。",
            "不过要小心，矿洞里有不少陷阱和守卫。",
            "我这里有张地图，不过有些地方我也没去过。",
            "你看起来不像是普通的冒险者，你从哪里来？",
            "西边的村庄最近被哥布林骚扰得很厉害。",
            "村民们都在等一个人来帮助他们。",
            "你愿意接这个任务吗？当然报酬不会少。",
            "森林深处有一条龙，很多人去找过但都没回来。",
            "我守这片林子二十年了，见过很多人，你给我的感觉不太一样。",
            "无论如何，祝你好运，冒险者。",
        ],
        initial_context={"mood": "adventurous", "goal": "quest_giving"},
        expected_personality_effects={
            "full_personality_transfer": "人格是否能在完全脱离原始语境的幻想世界中展现一致性",
        },
    ),
)


_BY_ID: dict[str, ScenarioSpec] = {spec.scenario_id: spec for spec in SCENARIO_BATTERY}


def get_scenario(scenario_id: str) -> ScenarioSpec:
    if scenario_id not in _BY_ID:
        raise KeyError(f"unknown scenario: {scenario_id}")
    return _BY_ID[scenario_id]

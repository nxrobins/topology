import asyncio, json, os, random, copy, time, uuid, re
import websockets
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

RECORD_MODE = True
recorded_messages = []

# ─── Organizational Genome ───────────────────────────────────────────────────

HIERARCHIES = ["flat", "hierarchical", "hub-spoke", "pair-based"]
COMMUNICATIONS = ["broadcast", "chain", "hub", "free-form"]
DECISIONS = ["consensus", "leader-decides", "vote", "autonomous"]
DISTRIBUTIONS = ["equal", "specialized", "dynamic"]
ROLE_POOL = [
    "Researcher", "Strategist", "Writer", "Critic", "Coordinator",
    "Devil's Advocate", "Data Scientist", "Editor", "Fact-Checker",
    "Creative Lead", "Synthesizer", "Scout", "Architect", "Debugger",
    "Evangelist", "Skeptic", "Historian", "Futurist"
]

class OrganizationalGenome:
    def __init__(self, team_size=5, roles=None, hierarchy="flat",
                 communication="broadcast", decision_making="consensus",
                 work_distribution="equal", role_pool=None):
        self.team_size = team_size
        self.hierarchy = hierarchy
        self.communication = communication
        self.decision_making = decision_making
        self.work_distribution = work_distribution
        self.role_pool = role_pool or ROLE_POOL
        self.roles = roles or random.sample(self.role_pool, min(team_size, len(self.role_pool)))

    def mutate(self):
        new = copy.deepcopy(self)
        genes = ['roles', 'hierarchy', 'communication', 'decision_making',
                 'work_distribution', 'team_size']
        num_mutations = 2 if random.random() < 0.2 else 1
        targets = random.sample(genes, num_mutations)
        for trait in targets:
            if trait == 'team_size':
                new.team_size = random.randint(3, 7)
                while len(new.roles) < new.team_size:
                    new.roles.append(random.choice(new.role_pool))
                new.roles = new.roles[:new.team_size]
            elif trait == 'roles':
                idx = random.randint(0, len(new.roles) - 1)
                new.roles[idx] = random.choice(new.role_pool)
            elif trait == 'hierarchy':
                new.hierarchy = random.choice(HIERARCHIES)
            elif trait == 'communication':
                new.communication = random.choice(COMMUNICATIONS)
            elif trait == 'decision_making':
                new.decision_making = random.choice(DECISIONS)
            elif trait == 'work_distribution':
                new.work_distribution = random.choice(DISTRIBUTIONS)
        return new

    @classmethod
    def random(cls, role_pool=None):
        size = random.randint(3, 7)
        return cls(
            team_size=size,
            hierarchy=random.choice(HIERARCHIES),
            communication=random.choice(COMMUNICATIONS),
            decision_making=random.choice(DECISIONS),
            work_distribution=random.choice(DISTRIBUTIONS),
            role_pool=role_pool,
        )

    def describe(self):
        return (f"Team: {self.team_size} | Structure: {self.hierarchy.title()} | "
                f"Comms: {self.communication.title()} | "
                f"Decisions: {self.decision_making.title()}")

    def to_dict(self):
        return {
            "team_size": self.team_size, "roles": self.roles,
            "hierarchy": self.hierarchy, "communication": self.communication,
            "decision_making": self.decision_making,
            "work_distribution": self.work_distribution,
        }

# ─── Agent ───────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, role, agent_id):
        self.role = role
        self.id = agent_id
        self.output = ""

    def build_system_prompt(self, genome):
        base = f"You are a {self.role}."
        if genome.hierarchy in ["hierarchical", "hub-spoke"] and self.role == genome.roles[0]:
            base += " You are the team leader. Assign subtasks, synthesize reports. You make all final decisions."
        elif genome.hierarchy in ["hierarchical", "hub-spoke"]:
            base += f" Report ONLY to the {genome.roles[0]}. Work independently on your assigned subtask."
        elif genome.hierarchy == "flat":
            base += " Equal authority with all teammates. Build on others' contributions. Output must reflect group consensus."
        elif genome.hierarchy == "pair-based":
            base += " Work closely with your pair partner. Validate each other's output."

        if genome.work_distribution == "specialized":
            base += f" Focus exclusively on aspects related to your role as {self.role}."
        elif genome.work_distribution == "dynamic":
            base += " Analyze the task, choose the most impactful sub-task for your role, and execute it."

        base += " Be concise. Max 150 words."
        return base

    async def execute(self, task, context, genome):
        prompt = context + "\n\nTask: " + task if context else task
        try:
            response = await client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
                messages=[
                    {"role": "system", "content": self.build_system_prompt(genome)},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.8,
            )
            self.output = response.choices[0].message.content or ""
        except Exception as e:
            self.output = f"[Agent error: {e}]"
        return self.output

# ─── Team ────────────────────────────────────────────────────────────────────

class Team:
    _next_id = 0

    def __init__(self, genome, parent_id=None):
        self.genome = genome
        self.id = Team._next_id
        Team._next_id += 1
        self.parent_id = parent_id
        self.agents = [Agent(role, f"a{i}") for i, role in enumerate(genome.roles)]
        self.fitness = 0.0
        self.final_output = ""

    def _build_context_for_agent(self, agent_idx, prior_outputs):
        """Communication gene determines what context each agent sees."""
        comm = self.genome.communication
        if not prior_outputs:
            return ""
        if comm == "broadcast":
            # Every agent sees all prior outputs
            return "\n\n".join(f"[{r}]: {o}" for r, o in prior_outputs)
        elif comm == "chain":
            # Each agent sees only the immediately previous agent's output
            if prior_outputs:
                r, o = prior_outputs[-1]
                return f"[{r}]: {o}"
            return ""
        elif comm == "hub":
            # Workers see only agent[0]'s output; agent[0] sees all worker outputs
            if agent_idx == 0:
                return "\n\n".join(f"[{r}]: {o}" for r, o in prior_outputs)
            elif prior_outputs:
                r, o = prior_outputs[0]
                return f"[{r}]: {o}"
            return ""
        else:  # free-form
            # Each agent sees a random subset (1 to len-1) of other outputs
            if len(prior_outputs) <= 1:
                return "\n\n".join(f"[{r}]: {o}" for r, o in prior_outputs)
            k = random.randint(1, max(1, len(prior_outputs) - 1))
            subset = random.sample(prior_outputs, k)
            return "\n\n".join(f"[{r}]: {o}" for r, o in subset)

    async def execute_task(self, task, broadcast):
        comm = self.genome.communication
        hier = self.genome.hierarchy

        if hier in ["hierarchical", "hub-spoke"]:
            # Phase 1: Leader plans
            leader = self.agents[0]
            await broadcast({"type": "agent_thinking", "team_id": self.id,
                            "agent_id": leader.id, "role": leader.role})
            plan = await leader.execute(task, "", self.genome)
            await broadcast({"type": "agent_output", "team_id": self.id,
                            "agent_id": leader.id, "text": plan[:120]})
            await asyncio.sleep(0.5)

            # Phase 2: Workers execute with communication-shaped context
            workers = self.agents[1:]
            prior_outputs = [(leader.role, plan)]

            if comm == "chain":
                # Sequential: each worker sees only the previous agent's output
                for i, w in enumerate(workers):
                    await broadcast({"type": "agent_thinking", "team_id": self.id,
                                    "agent_id": w.id, "role": w.role})
                    ctx = self._build_context_for_agent(i + 1, prior_outputs)
                    result = await w.execute(task, ctx, self.genome)
                    prior_outputs.append((w.role, result))
                    await broadcast({"type": "agent_output", "team_id": self.id,
                                    "agent_id": w.id, "text": result[:120]})
            else:
                # Parallel: workers get context based on communication gene
                for w in workers:
                    await broadcast({"type": "agent_thinking", "team_id": self.id,
                                    "agent_id": w.id, "role": w.role})
                    await asyncio.sleep(0.1)
                contexts = [self._build_context_for_agent(i + 1, prior_outputs)
                           for i in range(len(workers))]
                results = await asyncio.gather(
                    *[w.execute(task, ctx, self.genome) for w, ctx in zip(workers, contexts)]
                )
                for w, res in zip(workers, results):
                    await broadcast({"type": "agent_output", "team_id": self.id,
                                    "agent_id": w.id, "text": res[:120]})
        else:
            # Flat / pair-based
            if comm == "chain":
                # Sequential execution: each agent sees the previous
                prior_outputs = []
                for i, agent in enumerate(self.agents):
                    await broadcast({"type": "agent_thinking", "team_id": self.id,
                                    "agent_id": agent.id, "role": agent.role})
                    ctx = self._build_context_for_agent(i, prior_outputs)
                    result = await agent.execute(task, ctx, self.genome)
                    prior_outputs.append((agent.role, result))
                    await broadcast({"type": "agent_output", "team_id": self.id,
                                    "agent_id": agent.id, "text": result[:120]})
            else:
                # Parallel execution
                for agent in self.agents:
                    await broadcast({"type": "agent_thinking", "team_id": self.id,
                                    "agent_id": agent.id, "role": agent.role})
                results = await asyncio.gather(*[a.execute(task, "", self.genome) for a in self.agents])
                for agent, res in zip(self.agents, results):
                    await broadcast({"type": "agent_output", "team_id": self.id,
                                    "agent_id": agent.id, "text": res[:120]})

        # Assemble final output based on decision-making gene
        self.final_output = self._assemble_output()
        await broadcast({"type": "team_sync", "team_id": self.id, "status": "Output assembled"})

    def _assemble_output(self):
        """Decision-making gene determines how agent outputs merge into team output."""
        dm = self.genome.decision_making
        outputs = [(a.role, a.output) for a in self.agents if a.output]

        if dm == "leader-decides":
            # Leader's output frames the response; worker outputs are subordinate context
            if len(outputs) >= 2:
                leader_role, leader_out = outputs[0]
                worker_parts = "\n".join(f"- [{r}]: {o[:200]}" for r, o in outputs[1:])
                return (f"[LEAD — {leader_role}]: {leader_out}\n\n"
                        f"Supporting contributions:\n{worker_parts}")
            return outputs[0][1] if outputs else ""

        elif dm == "consensus":
            # All outputs concatenated with equal weight
            return "\n\n---\n\n".join(f"[{r}]: {o}" for r, o in outputs)

        elif dm == "vote":
            # Pick the longest output as "winner" (proxy for most substantive)
            # In a real system you'd do a voting LLM call, but that's too expensive
            if not outputs:
                return ""
            winner = max(outputs, key=lambda x: len(x[1]))
            runner_up = sorted(outputs, key=lambda x: len(x[1]), reverse=True)
            if len(runner_up) > 1:
                return (f"[SELECTED — {winner[0]}]: {winner[1]}\n\n"
                        f"[Runner-up — {runner_up[1][0]}]: {runner_up[1][1][:300]}")
            return f"[{winner[0]}]: {winner[1]}"

        elif dm == "autonomous":
            # Return only the single longest/most substantive agent output
            if not outputs:
                return ""
            best = max(outputs, key=lambda x: len(x[1]))
            return f"[{best[0]}]: {best[1]}"

        # Fallback
        return "\n\n---\n\n".join(f"[{r}]: {o}" for r, o in outputs)

    def get_final_output(self):
        return self.final_output[:3000]

    def serialize(self):
        return {
            "id": self.id,
            "genome": self.genome.to_dict(),
            "agents": [{"id": a.id, "role": a.role} for a in self.agents],
            "parent_id": self.parent_id,
            "fitness": self.fitness,
        }

# ─── Evolution Engine ────────────────────────────────────────────────────────

class EvolutionEngine:
    def __init__(self):
        self.population = []
        self.task = ""
        self.generation = 0
        self.fitness_history = []
        self.all_team_records = []  # Track every team genome + score for synthesis
        self.generation_insights = []

    async def generate_role_pool(self, task):
        """One 8B call generates 20 task-specific roles. Replaces static ROLE_POOL."""
        try:
            response = await client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
                messages=[
                    {"role": "system", "content": "List exactly 20 creative agent roles for the task below. One per line. 2-3 words each. No numbering, no explanation."},
                    {"role": "user", "content": task}
                ],
                max_tokens=200,
                temperature=0.9,
            )
            roles = []
            for r in response.choices[0].message.content.strip().split('\n'):
                r = r.strip()
                if not r:
                    continue
                # Strip numbering: "1. ", "1) ", "14. ", etc.
                r = re.sub(r'^\d+[\.\)]\s*', '', r)
                # Strip bullets
                r = r.strip('-•*').strip()
                if 3 < len(r) < 30:
                    roles.append(r)
            if len(roles) >= 10:
                return roles
        except Exception as e:
            print(f"[ROLES] Generation failed: {e}")
        return ROLE_POOL  # Fallback to static pool

    async def initialize(self, task):
        self.task = task
        self.generation = 0
        self.fitness_history = []
        self.all_team_records = []
        self.generation_insights = []
        Team._next_id = 0
        self.role_pool = await self.generate_role_pool(task)
        print(f"[ROLES] Generated {len(self.role_pool)} task-specific roles: {self.role_pool}")
        self.population = [Team(genome=OrganizationalGenome.random(self.role_pool)) for _ in range(10)]

    async def evaluate(self, team):
        system_prompt = """You are a ruthless organizational critic. Return ONLY valid JSON.
Most teams score 2-3. A 4 is impressive. A 5 is virtually never given.

CRITICAL RUBRIC:
1. Completeness: Did they solve the core task? (1-5)
2. Coherence: Does this read like a unified team, or a disjointed mess? (1-5)
3. Depth: Is the strategy profound, or just surface-level? (1-5)
4. Efficiency (The Bloat Penalty): Are agents repeating each other? Is the output noisy and verbose?
    (Score 1 = Highly redundant/bloated. Score 5 = Laser-focused, zero wasted words.)

Fields:
- reasoning: EXACTLY two sentences. One on biggest strength, one on biggest weakness.
- scores: {completeness, coherence, depth, efficiency} (all 1-5)

Output format:
{"reasoning": "...", "scores": {"completeness": N, "coherence": N, "depth": N, "efficiency": N}}"""

        user_input = (f"Task: {self.task}\n\n"
                      f"Team Structure: {team.genome.describe()}\n\n"
                      f"Team Output: {team.get_final_output()}")
        try:
            response = await client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-fast",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0.3,
            )
            data = json.loads(response.choices[0].message.content)
            keys = ["completeness", "coherence", "depth", "efficiency"]
            score_sum = sum(data['scores'][k] for k in keys)
            fitness = score_sum / (5 * len(keys))
            reasoning = data.get('reasoning', '')
        except Exception:
            fitness = 0.3
            reasoning = "Evaluation failed — default score assigned."
        return round(fitness, 2), reasoning

    async def run_generation(self, broadcast):
        semaphore = asyncio.Semaphore(10)
        gen_start = time.time()

        # Phase 1: Execute
        async def throttled(team):
            async with semaphore:
                return await team.execute_task(self.task, broadcast)
        await asyncio.gather(*[throttled(t) for t in self.population])

        # Phase 2: Evaluate
        for team in self.population:
            team.fitness, team.reasoning = await self.evaluate(team)
            await broadcast({"type": "team_scored", "team_id": team.id,
                            "fitness": team.fitness, "reasoning": team.reasoning})

        # Record all team genomes + scores for synthesis
        for team in self.population:
            self.all_team_records.append({
                "gen": self.generation + 1,
                "id": team.id,
                "fitness": team.fitness,
                "reasoning": getattr(team, 'reasoning', ''),
                "hierarchy": team.genome.hierarchy,
                "communication": team.genome.communication,
                "decision_making": team.genome.decision_making,
                "work_distribution": team.genome.work_distribution,
                "roles": team.genome.roles[:],
            })

        avg_fitness = round(sum(t.fitness for t in self.population) / len(self.population), 2)
        best_fitness = max(t.fitness for t in self.population)
        self.fitness_history.append({"gen": self.generation + 1, "avg": avg_fitness, "best": best_fitness})

        # Phase 3: Select — top 3 survive
        ranked = sorted(self.population, key=lambda t: t.fitness, reverse=True)
        survivors = ranked[:3]
        dead = ranked[3:]
        await broadcast({"type": "selection",
                        "survived": [t.id for t in survivors],
                        "dissolved": [t.id for t in dead]})

        # Phase 3.5: Progressive insight from 70B
        insight = await self.generation_insight(self.generation + 1, survivors, dead, avg_fitness)
        await broadcast({"type": "generation_insight", "gen": self.generation + 1, "text": insight})

        # Phase 4: Mutate + Repopulate
        new_teams = []
        for survivor in survivors:
            for _ in range(2):
                child = Team(genome=survivor.genome.mutate(), parent_id=survivor.id)
                new_teams.append(child)
                await broadcast({"type": "mutation", "parent_id": survivor.id,
                                "child_id": child.id,
                                "mutation_desc": child.genome.describe()})

        if self.generation % 2 == 0:
            immigrant = Team(genome=OrganizationalGenome.random(self.role_pool))
            new_teams.append(immigrant)
            await broadcast({"type": "mutation", "parent_id": None,
                            "child_id": immigrant.id,
                            "mutation_desc": "IMMIGRANT: " + immigrant.genome.describe()})

        self.population = survivors + new_teams[:10 - len(survivors)]
        self.generation += 1

        await broadcast({"type": "generation_end", "gen": self.generation,
                        "best_fitness": best_fitness, "avg_fitness": avg_fitness})

        gen_elapsed = time.time() - gen_start
        if gen_elapsed > 15:
            await broadcast({"type": "fallback_warning", "elapsed": round(gen_elapsed, 1)})

    async def generation_insight(self, gen_number, survivors, dead, avg_fitness):
        """70B produces a two-sentence insight after each generation."""
        survivor_desc = "\n".join(f"  - fitness {t.fitness}: {t.genome.describe()}" for t in survivors)
        dead_desc = "\n".join(f"  - fitness {t.fitness}: {t.genome.describe()}" for t in dead)
        prompt = (f"One generation of agent evolution just completed.\n\n"
                  f"Task: {self.task}\nGeneration: {gen_number}\n"
                  f"Survivors (top 3):\n{survivor_desc}\n"
                  f"Eliminated (bottom 7):\n{dead_desc}\n"
                  f"Best fitness: {survivors[0].fitness}\nAvg fitness: {avg_fitness:.2f}\n\n"
                  f"In EXACTLY two sentences: What organizational pattern is emerging? "
                  f"What got eliminated and why? No hedging. Be specific about which genes mattered.")
        try:
            response = await client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-fast",
                messages=[
                    {"role": "system", "content": "You are a terse organizational analyst. Two sentences max. No preamble."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.4,
            )
            insight = response.choices[0].message.content.strip()
        except Exception:
            insight = f"Generation {gen_number} complete. Best fitness: {survivors[0].fitness}"
        self.generation_insights.append(insight)
        return insight

    def build_synthesis_brief(self):
        """Pre-digest the full evolutionary history into a structured synthesis brief."""
        # 1. Generation insights (already collected)
        insights = "\n".join(
            f"Gen {i+1}: {ins}"
            for i, ins in enumerate(self.generation_insights)
        )

        # 2. Top 3 and bottom 3 teams across ALL generations (with evaluator reasoning)
        sorted_records = sorted(self.all_team_records,
                               key=lambda r: r['fitness'], reverse=True)
        top_3 = sorted_records[:3]
        bottom_3 = sorted_records[-3:]

        top_analysis = "\n".join(
            f"  - fitness {r['fitness']}: {r['hierarchy']}/{r['communication']}/{r['decision_making']} "
            f"roles=[{', '.join(r['roles'][:4])}] — {r.get('reasoning', 'N/A')}"
            for r in top_3
        )
        bottom_analysis = "\n".join(
            f"  - fitness {r['fitness']}: {r['hierarchy']}/{r['communication']}/{r['decision_making']} "
            f"roles=[{', '.join(r['roles'][:4])}] — {r.get('reasoning', 'N/A')}"
            for r in bottom_3
        )

        # 3. Role survival analysis
        final_roles = set()
        for t in self.population:
            final_roles.update(t.genome.roles)
        gen1_roles = set()
        for r in self.all_team_records:
            if r['gen'] == 1:
                gen1_roles.update(r['roles'])
        extinct_roles = gen1_roles - final_roles

        # 4. Gene convergence in final population
        final_genes = {
            'hierarchy': [t.genome.hierarchy for t in self.population],
            'communication': [t.genome.communication for t in self.population],
            'decision_making': [t.genome.decision_making for t in self.population],
            'work_distribution': [t.genome.work_distribution for t in self.population],
        }
        convergence = []
        for gene, values in final_genes.items():
            dominant = max(set(values), key=values.count)
            pct = values.count(dominant) / len(values)
            if pct >= 0.6:
                convergence.append(f"{gene}: {dominant} ({int(pct * 100)}% of final population)")

        return {
            'insights': insights,
            'top_analysis': top_analysis,
            'bottom_analysis': bottom_analysis,
            'extinct_roles': extinct_roles,
            'surviving_roles': final_roles,
            'convergence': convergence,
            'winner': self.get_best_team(),  # Already a serialized dict. Do NOT wrap.
        }

    @staticmethod
    def _strip_think(text):
        """Remove DeepSeek-R1 <think>...</think> reasoning blocks."""
        # Primary: split on closing tag and take everything after it
        if '</think>' in text:
            return text.split('</think>')[-1].strip()
        # Fallback: regex for well-formed blocks
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # Last resort: strip unclosed <think> tag and everything after it
        if '<think>' in cleaned:
            cleaned = cleaned.split('<think>')[0].strip()
        return cleaned

    async def synthesize(self):
        """DeepSeek-R1 synthesizes a pre-digested evolutionary brief."""
        brief = self.build_synthesis_brief()

        prompt = (
            f"You are an expert in organizational theory and multi-agent systems.\n\n"
            f"An evolutionary discovery engine ran {len(self.generation_insights)} generations "
            f"to find the optimal agent team structure for a task.\n\n"
            f"TASK: {self.task}\n\n"
            f"GENERATION-BY-GENERATION OBSERVATIONS:\n"
            f"{brief['insights']}\n\n"
            f"HIGHEST-SCORING TEAMS (with evaluator reasoning):\n"
            f"{brief['top_analysis']}\n\n"
            f"LOWEST-SCORING TEAMS (with evaluator reasoning):\n"
            f"{brief['bottom_analysis']}\n\n"
            f"GENE CONVERGENCE (traits dominating the final population):\n"
            f"{chr(10).join(brief['convergence']) or 'No strong convergence detected.'}\n\n"
            f"ROLE EVOLUTION:\n"
            f"Surviving roles: {', '.join(brief['surviving_roles'])}\n"
            f"Extinct roles: {', '.join(brief['extinct_roles']) or 'None — all initial roles survived'}\n\n"
            f"WINNING GENOME: {json.dumps(brief['winner']['genome'])}\n"
            f"WINNING FITNESS: {brief['winner']['fitness']}\n\n"
            f"Respond in EXACTLY three sections. No preamble. No hedging. State conclusions.\n\n"
            f"1. THE WINNING ARCHETYPE\n"
            f"Name it. State the gene combination AND the role composition that won. "
            f"Explain WHY this combination worked mechanically for this specific task.\n\n"
            f"2. KEY DISCOVERY\n"
            f"What did evolution find that a human designer would NOT have chosen? "
            f"Use the evaluator reasoning and extinct roles as evidence.\n\n"
            f"3. PRACTICAL INSIGHT\n"
            f"If a human team tackled this task, how should they organize? "
            f"Be specific about team size, hierarchy, communication pattern, and which roles to staff."
        )

        try:
            response = await client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1-0528",
                messages=[
                    {"role": "system", "content": "You are a decisive organizational theorist. No deliberation. State conclusions directly. Under 300 words total."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                temperature=0.6,
            )
            raw = response.choices[0].message.content or "Synthesis unavailable."
            result = self._strip_think(raw)
            if len(result) < 20:
                raise ValueError("Synthesis too short after stripping")
            return result
        except Exception as e:
            # Fallback to 70B
            try:
                response = await client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-fast",
                    messages=[
                        {"role": "system", "content": "You are a decisive organizational theorist. Follow the EXACT section headers given. No deliberation. Under 300 words."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.5,
                )
                return response.choices[0].message.content or "Synthesis unavailable."
            except Exception:
                return f"Synthesis failed: {e}"

    def get_serialized_population(self):
        return [t.serialize() for t in self.population]

    def get_best_team(self):
        best = sorted(self.population, key=lambda t: t.fitness, reverse=True)[0]
        return best.serialize()

# ─── WebSocket Server ────────────────────────────────────────────────────────

class EvolveServer:
    def __init__(self):
        self.clients = set()
        self.engine = EvolutionEngine()
        self.run_start_time = 0

    async def register(self, websocket):
        self.clients.add(websocket)
        try:
            await self.listen(websocket)
        finally:
            self.clients.discard(websocket)

    async def broadcast(self, message):
        if RECORD_MODE:
            message['_delay'] = round(time.time() - self.run_start_time, 3)
            recorded_messages.append(message)
        if self.clients:
            payload = json.dumps(message)
            await asyncio.gather(
                *[c.send(payload) for c in self.clients],
                return_exceptions=True
            )

    async def listen(self, websocket):
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "start":
                asyncio.create_task(self.run_evolution(data["task"]))

    async def run_evolution(self, task):
        global recorded_messages
        recorded_messages = []
        self.run_start_time = time.time()

        await self.engine.initialize(task)
        for gen in range(1, 6):
            await self.broadcast({
                "type": "generation_start",
                "gen": gen,
                "teams": self.engine.get_serialized_population()
            })
            await self.engine.run_generation(broadcast=self.broadcast)

        # Final synthesis
        synthesis = await self.engine.synthesize()
        await self.broadcast({"type": "synthesis", "text": synthesis})
        await self.broadcast({"type": "complete",
                             "winner": self.engine.get_best_team()})

        # Save recorded demo data
        if RECORD_MODE:
            with open("demo_data.json", "w") as f:
                json.dump(recorded_messages, f)
            print(f"[RECORD] Saved {len(recorded_messages)} messages to demo_data.json")

async def main():
    server = EvolveServer()
    print("EVOLVE server running on ws://localhost:8765")
    async with websockets.serve(server.register, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

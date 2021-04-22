# trying to simulate game-theory dynamics of a system of agents where defecting is disincentivized and egalitarian cooperation is a STABLE equilibrium


import random
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, name, production_skill, goods_endowment, needs_by_good, forgiveness):
        self.name = name
        self.production_skill = production_skill
        self.inventory = goods_endowment
        self.needs = needs_by_good
        self.forgiveness = forgiveness  # more forgiving means you will still trade more with people who defect
        self.energy = 10
        self.ledger = {}
        self.alive = True

    def __repr__(self):
        return f"<Agent {self.name}>"

    @staticmethod
    def random(name, goods):
        production_skill = {}
        endowment = {}
        needs = {}
        for good in goods:
            production_skill[good] = 1+np.random.lognormal(1,5)
            endowment[good] = 1+np.random.lognormal(1,5)
            needs[good] = 1+np.random.lognormal(1,5)
        forgiveness = 1+np.random.lognormal(1,5)
        return Agent(name, production_skill, endowment, needs, forgiveness)

    def get_marginal_utility(self, good):
        current = self.inventory[good]
        need = self.needs[good]
        u = below_need_constant_then_exp_marginal_utility(current, need)
        return u

    def get_price(self, g0, g1):
        # return price of g0 in terms of g1
        # e.g. price(corn, wheat) will give the number of wheats needed to buy a corn
        mar_u0 = self.get_marginal_utility(g0)
        mar_u1 = self.get_marginal_utility(g1)
        # e.g. marginal utility of corn is 2 utils per corn, marginal utility of wheat is 6 utils per wheat
        # so then the price of corn in terms of wheat should be 1/3, each corn is worth 1/3 of a wheat to this agent
        return mar_u0 / mar_u1

    def get_marginal_utilities(self, goods):
        d = {}
        for g in goods:
            mar_u = self.get_marginal_utility(g)
            d[g] = mar_u
        return d

    def list_marginal_utilities(self, goods):
        d = self.get_marginal_utilities(goods)
        for k, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{self} has MU {v:.6f} for {k}")

    def report_inventory_and_needs(self, goods):
        for g in goods:
            inv = self.inventory[g]
            need = self.needs[g]
            print(f"{self} has {inv:.2f} and needs {need:.2f} units of {g} (ratio {inv/need:.4f})")

    def get_most_needed_good(self, goods):
        return max(self.get_marginal_utilities(goods).items(), key=lambda kv: kv[1])

    def get_least_needed_good(self, goods):
        return min(self.get_marginal_utilities(goods).items(), key=lambda kv: kv[1])
        
    def request_from(self, other, goods, other_agents):
        self_wanted = self.get_most_needed_good(goods)
        other_wanted = other.get_most_needed_good(goods)
        g0 = self_wanted
        g1 = other_wanted
        price_to_self = self.get_price(g0, g1)
        price_to_other = other.get_price(g0, g1)
        print(f"g0 {g0}; g1 {g1}; p0/1 to agent0 = {price_to_self}; p0/1 to agent1 = {price_to_other}")

    def find_good(self, goods, agents):
        # g, mar_u = self.get_most_needed_good(goods)
        potentially_needed_goods = [g for g in goods]
        while True:
            if len(potentially_needed_goods) == 0:
                print(f"{self} has everything they need")
                return
            g = random.choice(potentially_needed_goods)
            mar_u = self.get_marginal_utility(g)
            amount = self.needs[g] - self.inventory[g]
            if amount <= 0:
                potentially_needed_goods.remove(g)
            else:
                break

        print(f"{self} is in search of {amount:.2f} units of {g} with marginal utility {mar_u:.4f}")

        candidate_producers = [a for a in agents]
        while len(candidate_producers) > 0:
            a = get_agent_who_produces_good(g, candidate_producers)
            if a is self:
                self.produce(g, amount)
                print(f"{self} produced {amount:.2f} units of {g} themself")
                return
            else:
                if a.will_do_favor_for(self):
                    amount_produced, energy_expended = a.produce(g, amount)
                    print(f"{a} has produced {amount:.2f} units of {g} for {self}")
                    utility_gotten = amount_produced * self.get_marginal_utility(g)
                    a.record_favor_for(self, energy_expended)
                    self.record_favor_from(a, utility_gotten)
                    return
                else:
                    print(f"{a} refuses to do a favor for {self}")
                    candidate_producers.remove(a)
        # if got here
        print(f"{self} failed to find anyone willing to produce {g}")

    def record_favor_for(self, other, energy_expended):
        if other not in self.ledger:
            self.ledger[other] = 0
        self.ledger[other] -= energy_expended
        print(f"{self} records expending {energy_expended:.2f} for {other}, ledger now {self.ledger[other]:.2f}")

    def record_favor_from(self, other, utility_gotten):
        if other not in self.ledger:
            self.ledger[other] = 0
        self.ledger[other] += utility_gotten
        print(f"{self} records receiving {utility_gotten:.2f} from {other}, ledger now {self.ledger[other]:.2f}")

    def will_do_favor_for(self, other):
        x = self.ledger.get(other, 0)
        prob = min(1, np.exp(x / self.forgiveness))  # if positive relationship, trade with 100% probability
        return random.random() < prob

    def consume(self, goods, proportion):
        # consuming `need_vector*proportion` will give you `proportion units` of energy
        need_units_in_inventory = {g: self.inventory[g]/self.needs[g] for g in goods}
        min_need_units = min(need_units_in_inventory.values())
        proportion = min(proportion, min_need_units)  # can't consume more than you have, if there is some limiting good
        print(f"{self} can consume {proportion:.2f} of their needs")

        amounts_consumed = {g: proportion*self.needs[g] for g in goods}

        for g in goods:
            assert float_leq(amounts_consumed[g],self.inventory[g]), f"{self} cannot consume more {g} than they have (tried to consume {amounts_consumed[g]} but have {self.inventory[g]})"
        for g in goods:
            self.inventory[g] -= amounts_consumed[g]
        self.energy += proportion
        print(f"{self} now has energy {self.energy:.2f}")
        return proportion

    def produce(self, good, amount):
        energy_expended = amount / self.production_skill[good]
        self.energy -= energy_expended
        print(f"{self} expended {energy_expended:.2f} by producing {amount:.2f} units of {good}")
        print(f"{self} now has energy {self.energy:.2f}")
        return amount, energy_expended

    def pass_day(self, goods):
        self.consume(goods, 10)
        self.energy -= 1
        print(f"{self} passed a day, now has energy {self.energy:.2f}")
        if self.energy <= 0:
            self.die()

    def die(self):
        print(f"{self} has died")
        self.alive = False


def float_leq(a,b):
    return a<b or abs(a-b) < 1e-8


def get_agent_who_produces_good(good, agents):
    # random weighted by production skill
    d = {a: a.production_skill[good] for a in agents}
    total_skill = sum(d.values())
    return np.random.choice(agents, p=[d[a]/total_skill for a in agents])


def below_need_constant_then_exp_marginal_utility(amount_had, amount_needed):
    # scale up for lower need
    x = amount_had
    a = amount_needed
    return 1/(1+a) * min(1, np.exp(1 - x/(1+a)))


def below_need_constant_then_linear_marginal_utility(amount_had, amount_needed):
    x = amount_had
    a = amount_needed
    # don't let it go negative, will cause problems for weighted choice
    return 1/(1+a) * min(1, max(0, 2 - x/(1+a)))


def below_need_constant_then_zero_marginal_utility(amount_had, amount_needed):
    x = amount_had
    a = amount_needed
    return 1/(1+a) if x <= a else 0



if __name__ == "__main__":
    goods = ["water", "salt"] #, "grain", "metal", "wood", "meat", "fire"]
    n_agents = 100
    agents = []
    for i in range(n_agents):
        name = f"{i}"
        agent = Agent.random(name, goods)
        agents.append(agent)

    n_days = 100
    alive_agents = [a for a in agents]
    for day in range(n_days):
        print(f"\nDay {day}")
        if len(alive_agents) == 0:
            print("everyone is dead")
            break
        for i in range(10 * n_agents):
            print("\n-- begin new transaction")
            a = random.choice(alive_agents)
            a.find_good(goods, alive_agents)
            a.report_inventory_and_needs(goods)
            # input("press enter to continue")
        print("\n-- begin consumption period")
        for a in alive_agents:
            a.pass_day(goods)
        alive_agents = [a for a in agents if a.alive]
        print(f"Day {day} is now over. {len(alive_agents)}/{len(agents)} people are alive.")

    # TODO make all trading one direction only, over time people accumulate social trust amount
    # if someone becomes less trusted (has reputation as a defector), people will be less likely to trade with them at all (won't raise prices, will just refuse to interact, cutting them out of the social network)


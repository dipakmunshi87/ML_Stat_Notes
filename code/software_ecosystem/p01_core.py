from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Iterator

@dataclass(order=True)
class Star:
    mass: float                       # solar masses
    name: str = field(compare=False)

    @property
    def luminosity(self) -> float:    # rough main-sequence scaling L ~ M^3.5
        return self.mass ** 3.5

def stream(path_lines: list[str]) -> Iterator[Star]:
    """Generator: parse lines lazily, skipping comments."""
    for line in path_lines:
        if line.startswith("#"):
            continue
        name, m = line.split(",")
        yield Star(float(m), name.strip())

lines = ["# name, mass", "Sun, 1.0", "Sirius A, 2.06", "Proxima, 0.12", "Vega, 2.14"]
stars = sorted(stream(lines), reverse=True)            # order=True: sort by mass
print([s.name for s in stars])
print({s.name: round(s.luminosity, 2) for s in stars})  # dict comprehension
by_class = defaultdict(list)
for s in stars:
    by_class["massive" if s.mass > 1.5 else "light"].append(s.name)
print(dict(by_class))
print(Counter("mississippi").most_common(2))
match stars[0]:                                         # structural pattern matching (3.10+)
    case Star(mass=m, name=n) if m > 2:
        print(f"{n} is heavy: {m=}")                    # f-string '=' debugging (3.8+)

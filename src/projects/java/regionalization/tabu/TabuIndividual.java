package regionalization.tabu;

import java.util.List;

public abstract interface TabuIndividual {
	public abstract double getValue();
	public abstract TabuIndividual applyMove( TabuMove tm );
	public abstract TabuMove getRandomMove();
	public abstract List<TabuMove> getAllMoves();
}

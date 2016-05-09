package heuristics.tabu;

public interface TabuMove<T> {
	public TabuMove<T> getInverse();
	public Object getAttribute();
}

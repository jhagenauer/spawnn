package inc_llm;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/* Bidirectional connections, could be used for any growing self-organizing network.
 * Currently only used for IncLLM.
 * Beware to always manage A-B and B-A 
 */
public class ConnectionsIncLLM {
	Map<double[], Map<double[], int[]>> cons = new HashMap<>();
	Set<int[]> age = new HashSet<int[]>();
	
	public void add(double[] a, double[] b) {
		if (!cons.containsKey(a))
			cons.put(a, new HashMap<double[], int[]>());
		if (cons.get(a).containsKey(b)) // update age
			cons.get(a).get(b)[0] = 0;
		else // add new
			cons.get(a).put(b, new int[] { 0 });

		if (!cons.containsKey(b))
			cons.put(b, new HashMap<double[], int[]>());
		if (cons.get(b).containsKey(a)) // update age
			cons.get(b).get(a)[0] = 0;
		else // add new
			cons.get(b).put(a, new int[] { 0 });
	}

	public int remove(double[] a, double[] b) {
		int[] r = cons.get(a).remove(b);
		cons.get(b).remove(a); // also returns r

		if (cons.get(a).isEmpty())
			cons.remove(a);

		if (cons.get(b).isEmpty())
			cons.remove(b);

		return r[0];
	}

	public Set<double[]> getNeighbors(double[] d, int depth) {
		Set<double[]> l = new HashSet<double[]>();
		if (depth == 0)
			return l;
		for (double[] nb : cons.get(d).keySet()) {
			if (!l.contains(nb)) {
				l.add(nb);
				l.addAll(getNeighbors(nb, depth - 1));
			}
		}
		return l;
	}

	public void increase() {
		for (Map<double[], int[]> v : cons.values())
			for (int[] a : v.values())
				a[0]++;
			
	}

	public void purge(int max) {
		Set<double[]> toRemove2 = new HashSet<double[]>();
		for (Entry<double[], Map<double[], int[]>> a : cons.entrySet()) {
			Set<double[]> toRemove = new HashSet<double[]>();
			for (Entry<double[], int[]> e : a.getValue().entrySet())
				if (e.getValue()[0] > max)
					toRemove.add(e.getKey());
			a.getValue().keySet().removeAll(toRemove);
			if (a.getValue().isEmpty())
				toRemove2.add(a.getKey());
		}
		cons.keySet().removeAll(toRemove2);
	}

	public void incAndPurge(int max) {
		Set<double[]> toRemove2 = new HashSet<double[]>();
		for (Entry<double[], Map<double[], int[]>> a : cons.entrySet()) {
			Set<double[]> toRemove = new HashSet<double[]>();
			for (Entry<double[], int[]> e : a.getValue().entrySet()) {
				e.getValue()[0]++;
				if (e.getValue()[0] > max)
					toRemove.add(e.getKey());
			}
			a.getValue().keySet().removeAll(toRemove);
			if (a.getValue().isEmpty())
				toRemove2.add(a.getKey());
		}
		cons.keySet().removeAll(toRemove2);
	}

	Set<double[]> getVertices() { // since it is bidirectional and double-connected, return just the keys is enough
		return cons.keySet();
	}
}

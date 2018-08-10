package context.space.binary_field;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.GridPos;
import spawnn.utils.DataUtils;

public class SpaceTestDiscrete {

	private static Logger log = Logger.getLogger(SpaceTestDiscrete.class);

	public static void main(String[] args) {
		Random r = new Random();
		final int maxDist = 5;
		//List<double[]> samples = DataUtils.readCSV("data/somsd/grid100x100.csv");
		//Map<double[], Map<double[], Double>> dMap = readDistMap(samples, "data/somsd/grid100x100.wtg");
		
		List<double[]> samples = DataUtils.readCSV("output/toroid50x50_0.csv");
		Map<double[], Map<double[], Double>> dMap = SpaceTestDiscrete.readDistMap(samples, "output/toroid50x50_0.wtg");
		
		// time test
		long time = System.currentTimeMillis();
		for (int i = 1; i < 20000; i++) {
			double[] d = samples.get(r.nextInt(samples.size()));
			//Set<double[]> s1 = getSurrounding(d, dMap, maxDist);
			Set<double[]> s2 = getSurroundingOld2(d, dMap, maxDist);
		}
		log.debug("took: " + (System.currentTimeMillis() - time));
		System.exit(1);
	}

	public static <T> Map<T, Set<Grid2D_Map<Boolean>>> getReceptiveFields(List<double[]> samples, final Map<double[], Map<double[], Double>> dMap, Map<T, Set<double[]>> bmus, int maxDist, int rfMaxSize, int[] ga, int fa) {
		int xDim = 0, yDim = 0;
		Map<double[], Set<double[]>> surroundings = new HashMap<double[], Set<double[]>>();
		for (Set<double[]> s : bmus.values()) {
			for (double[] center : s) {
				Set<double[]> sur = getSurrounding(center, dMap, maxDist);
				for (double[] nb : dMap.get(center).keySet()) {
					xDim = Math.max(xDim, (int) Math.abs(center[ga[0]] - nb[ga[0]]));
					yDim = Math.max(yDim, (int) Math.abs(center[ga[1]] - nb[ga[1]]));
				}
				surroundings.put(center, sur);
			}
		}

		Map<T, Set<Grid2D_Map<Boolean>>> rcf = new HashMap<T, Set<Grid2D_Map<Boolean>>>();
		for (T p : bmus.keySet()) {
			Set<Grid2D_Map<Boolean>> s = new HashSet<Grid2D_Map<Boolean>>();

			for (double[] center : bmus.get(p)) { // for each sample
				Grid2D_Map<Boolean> bf = new Grid2D_Map<Boolean>(0, 0);
				for (double[] d : surroundings.get(center)) {
					int x = (int) (d[ga[0]] - center[ga[0]]);
					int y = (int) (d[ga[1]] - center[ga[1]]);

					if (x > maxDist)
						x -= xDim + 1;
					if (y > maxDist)
						y -= yDim + 1;
					if (x < -maxDist)
						x += xDim + 1;
					if (y < -maxDist)
						y += yDim + 1;

					bf.setPrototypeAt(new GridPos(x, y), d[fa] > 0);
				}
				if (bf.size() == rfMaxSize)
					s.add(bf); // add only complete fields
			}

			if (!s.isEmpty())
				rcf.put(p, s);
		}
		return rcf;
	}

	public static <T> Map<T, Grid2D_Map<Boolean>> getIntersectReceptiveFields(Map<T, Set<Grid2D_Map<Boolean>>> rcp, int idx, int[] ga) {
		Map<T, Grid2D_Map<Boolean>> r = new HashMap<T, Grid2D_Map<Boolean>>();

		for (T p : rcp.keySet()) {

			Grid2D_Map<Boolean> intersect = null;
			for (Grid2D_Map<Boolean> bf : rcp.get(p)) {
				if (intersect == null)
					intersect = new Grid2D_Map<Boolean>(bf.getGridMap());
				else {
					/*
					 * intersect.intersect(bf); public void intersect(BinaryGrid2D sf) { Set<GridPos> rm = new HashSet<GridPos>(); for( GridPos p : getPositions() ) { if( !sf.getPositions().contains(p) || getPrototypeAt(p) != sf.getPrototypeAt(p) ) rm.add(p); } for( GridPos p : rm ) getGridMap().remove(p); }
					 */
					// get intersect of bf and intersect, result to intersect
					Set<GridPos> rm = new HashSet<GridPos>();
					for (GridPos gp : intersect.getPositions()) {
						if (!bf.getPositions().contains(gp) || intersect.getPrototypeAt(gp) != bf.getPrototypeAt(gp))
							rm.add(gp);
					}
					for (GridPos gp : rm)
						intersect.getGridMap().remove(gp);
				}
			}

			// reduce to connected component
			GridPos c = new GridPos(0, 0);
			if (!intersect.getPositions().contains(c)) // nothing is connected to center
				intersect = new Grid2D_Map<Boolean>(0, 0);
			else {
				Set<GridPos> toExpand = new HashSet<GridPos>();
				Set<GridPos> visited = new HashSet<GridPos>();
				toExpand.add(c);
				while (!toExpand.isEmpty()) {

					Set<GridPos> nbs = new HashSet<GridPos>();
					for (GridPos gp : toExpand)
						nbs.addAll(intersect.getNeighbours(gp));

					visited.addAll(toExpand);
					toExpand.clear();

					nbs.removeAll(visited);
					toExpand.addAll(nbs);
				}

				Set<GridPos> s = new HashSet<GridPos>(intersect.getPositions());
				s.removeAll(visited);
				for (GridPos gp : s)
					intersect.getGridMap().remove(gp);
			}

			r.put(p, intersect);
		}
		return r;
	}

	@Deprecated
	public static Set<double[]> getSurroundingOld(double[] n, final Map<double[], Map<double[], Double>> dMap, double maxDist) {
		Set<double[]> s = new HashSet<double[]>();
		s.add(n);

		for (Entry<double[], Double> nb : dMap.get(n).entrySet()) {
			if (nb.getValue() <= maxDist)
				s.addAll(getSurroundingOld(nb.getKey(), dMap, maxDist - nb.getValue()));
		}
		return s;
	}

	private static class QueueEntry implements Comparable<QueueEntry> {
		double[] node;
		double dist;

		public QueueEntry(double[] node, double dist) {
			this.node = node;
			this.dist = dist;
		}

		@Override
		public int compareTo(QueueEntry o) {
			if (dist == o.dist)
				return 0;
			else if (dist < o.dist)
				return -1;
			else
				return +1;
		}

		@Override
		public boolean equals(Object o) {
			log.debug("AHA");
			System.exit(1);
			return node.equals( ((QueueEntry)o).node );
		}
	}

	// dijkstra
	public static Set<double[]> getSurrounding(double[] from, final Map<double[], Map<double[], Double>> dMap, double maxDist) {
		Set<double[]> openList = new HashSet<double[]>();
		Map<double[], Double> distMap = new HashMap<double[], Double>();
		PriorityQueue<QueueEntry> pq = new PriorityQueue<QueueEntry>();

		openList.add(from);
		distMap.put(from, 0.0);
		pq.add(new QueueEntry(from, 0.0));

		while (!openList.isEmpty()) {
			// find nearest
			QueueEntry entry = pq.poll();
			while ( !openList.contains(entry.node) // repoll, if entry not on openlist  
					|| entry.dist != distMap.get(entry.node)) { // or dist is outdated
				entry = pq.poll();
			}
			double[] curNode = entry.node;
			double curMinCost = entry.dist;

			openList.remove(curNode);

			if (curMinCost > maxDist) {
				Set<double[]> s = new HashSet<double[]>();
				for (Entry<double[], Double> d : distMap.entrySet())
					if (d.getValue() <= maxDist)
						s.add(d.getKey());
				return s;
			}

			for (Entry<double[], Double> nb : dMap.get(curNode).entrySet()) {
				double d = curMinCost + nb.getValue();

				if (!distMap.containsKey(nb.getKey()) 
						|| d < distMap.get(nb.getKey())) {
					distMap.put(nb.getKey(), d);
					openList.add(nb.getKey());
					pq.add( new QueueEntry(nb.getKey(), d));
				}
			}
		}
		return null;
	}

	public static Set<double[]> getSurroundingOld2(double[] from, final Map<double[], Map<double[], Double>> dMap, double maxDist) {
		Set<double[]> openList = new HashSet<double[]>();
		Map<double[], Double> distMap = new HashMap<double[], Double>();
		PriorityQueue<QueueEntry> pq = new PriorityQueue<QueueEntry>();

		openList.add(from);
		distMap.put(from, 0.0);
		pq.add(new QueueEntry(from, 0.0));

		while (!openList.isEmpty()) {

			// find nearest
			double curMinCost = Double.POSITIVE_INFINITY;
			double[] curNode = null;
			for (double[] v : openList) {
				if (distMap.get(v) < curMinCost) {
					curMinCost = distMap.get(v);
					curNode = v;
				}
			}
			openList.remove(curNode);

			if (curMinCost > maxDist) {
				Set<double[]> s = new HashSet<double[]>();
				for (Entry<double[], Double> d : distMap.entrySet())
					if (d.getValue() <= maxDist)
						s.add(d.getKey());
				return s;
			}

			for (Entry<double[], Double> nb : dMap.get(curNode).entrySet()) {
				double d = curMinCost + nb.getValue();

				if (!distMap.containsKey(nb.getKey()) || d < distMap.get(nb.getKey())) {
					distMap.put(nb.getKey(), d);
					openList.add(nb.getKey());
				}
			}
		}
		return null;
	}

	public static Map<double[], Map<double[], Double>> readDistMap(List<double[]> samples, String fn) {
		Map<double[], Map<double[], Double>> distMap = new HashMap<double[], Map<double[], Double>>();

		for (double[] d : samples)
			distMap.put(d, new HashMap<double[], Double>());

		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(fn));

			String line = br.readLine(); // skip header
			while ((line = br.readLine()) != null) {
				String[] s = line.split(",");

				double[] from = samples.get(Integer.parseInt(s[0]));
				double[] to = samples.get(Integer.parseInt(s[1]));
				double dist = Double.parseDouble(s[2]);

				if (!distMap.containsKey(from))
					distMap.put(from, new HashMap<double[], Double>());

				distMap.get(from).put(to, dist);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				br.close();
			} catch (Exception e) {
			}
		}

		return distMap;
	}
}

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import edu.uci.ics.jung.algorithms.shortestpath.DijkstraShortestPath;
import edu.uci.ics.jung.graph.DirectedSparseGraph;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;

public class TestGrid {

	public static void main(String[] args) {
		Random r = new Random();
		Grid2D<double[]> grid = new Grid2DHex<double[]>(12, 8);

		DirectedSparseGraph<GridPos, double[]> graph = new DirectedSparseGraph<GridPos, double[]>();
		for (GridPos gp : grid.getPositions()) {
			if (!graph.getVertices().contains(gp))
				graph.addVertex(gp);

			for (GridPos nb : grid.getNeighbours(gp)) {
				if (!graph.getVertices().contains(nb))
					graph.addVertex(nb);
				graph.addEdge(new double[] {}, gp, nb);
			}
		}
		
		List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
		Collections.sort(pos);
		
		while (true) {
			GridPos p1 = pos.get(r.nextInt(pos.size()));
			GridPos p2 = pos.get(r.nextInt(pos.size()));
			
			//p1 = new GridPos(2,0);
			//p2 = new GridPos(1,2);

			int d1 = grid.dist(p1, p2);
			DijkstraShortestPath<GridPos, double[]> dsp = new DijkstraShortestPath<GridPos, double[]>(graph);
			int d2 = dsp.getPath(p1, p2).size();

			if (d1 != d2) {
				System.err.println(p1 + "->" + p2 + ":::" + d1 + "!=" + d2);
				for( double[] edge : dsp.getPath(p1, p2) ) {
					System.out.println(graph.getEndpoints(edge));
				}
				System.exit(1);
			}
		}

	}

}

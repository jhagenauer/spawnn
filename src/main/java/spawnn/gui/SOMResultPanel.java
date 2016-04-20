package spawnn.gui;

import java.awt.CardLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.ListCellRenderer;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.graph.UndirectedSparseGraph;
import net.miginfocom.swing.MigLayout;
import spawnn.dist.Dist;
import spawnn.gui.ClusterDialogGrid.ClusterAlgorithm;
import spawnn.gui.DistanceDialog.DistMode;
import spawnn.gui.DistanceDialog.StatMode;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.utils.SomToolboxUtils;
import spawnn.som.utils.SomUtils;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.SpatialDataFrame;

public class SOMResultPanel extends ResultPanel<GridPos> {

	private static Logger log = Logger.getLogger(SOMResultPanel.class);
	private static final long serialVersionUID = -4518072006960672609L;

	private JComboBox<String> gridComboBox;
	private JComboBox gridModeComboBox;
	private JButton btnExpGrid;
	private GridPanel pnlGrid;
	private GraphPanel pnlGraph;
	private JPanel cards;

	private Grid2D<double[]> grid;
	private Graph<double[], double[]> graph;

	private List<double[]> samples;
	
	private static final String RANDOM = "Random", DISTANCE = "Distance...", CLUSTER = "Cluster...";
	private static final String GRID = "SOM", GRAPH = "Neurons (Geo)";
		
	private Object currentGridComboBoxItem;

	public SOMResultPanel(Frame parent, SpatialDataFrame orig, List<double[]> samples, Map<GridPos, Set<double[]>> bmus, Grid2D<double[]> grid, Dist<double[]> fDist, Dist<double[]> gDist, int[] fa, int[] ga, boolean wmc) {
		super(parent,orig,samples,bmus,new ArrayList<GridPos>(grid.getPositions()),fDist,gDist);
		this.grid = grid;

		setLayout(new MigLayout(""));
		
		gridComboBox = new JComboBox<String>();
		gridComboBox.addItem(RANDOM);
		gridComboBox.addItem(DISTANCE);
		gridComboBox.addItem(CLUSTER);
		
		gridComboBox.setRenderer(new ComboSeparatorsRendererString((ListCellRenderer<String>)gridComboBox.getRenderer()){        
		    @Override
			protected boolean addSeparatorAfter(JList list, String value, int index) {
		    	return CLUSTER.equals(value);
			}                                                                            
		});     
		
		Set<Integer> fas = new HashSet<Integer>();
		for( int i : fa )
			fas.add(i);
		for( int i = 0; i < orig.names.size(); i++ ) {
			String s = orig.names.get(i);
			if( fas.contains(i))
				s+="*";
			gridComboBox.addItem(s);
		}
		if( wmc )
			for( int i = 0; i < orig.names.size(); i++ ) {
				String s = orig.names.get(i);
				if( fas.contains(i))
					s+="*";
				s+= " (ctx)";
				gridComboBox.addItem(s);
			}
		gridComboBox.addActionListener(this);
		currentGridComboBoxItem = gridComboBox.getSelectedItem();

		gridModeComboBox = new JComboBox();
		gridModeComboBox.addItem(GRID);
		if (ga != null && ga.length == 2) {
			gridModeComboBox.addItem(GRAPH);
			gridModeComboBox.addActionListener(this);
			gridModeComboBox.setEnabled(true);
		} else {
			gridModeComboBox.setEnabled(false);
		}
		gridModeComboBox.setToolTipText("Set neural layout.");

		btnExpGrid = new JButton("Export grid...");
		btnExpGrid.addActionListener(this);

		graph = new UndirectedSparseGraph<double[], double[]>();
		for (GridPos gp : grid.getPositions()) {
			double[] a = grid.getPrototypeAt(gp);
			if (!graph.getVertices().contains(a))
				graph.addVertex(a);

			for (GridPos nb : grid.getNeighbours(gp)) {
				double[] b = grid.getPrototypeAt(nb);
				if (!graph.getVertices().contains(b))
					graph.addVertex(b);

				if (gDist != null)
					graph.addEdge(new double[] { fDist.dist(a, b), gDist.dist(a, b) }, a, b);
				else
					graph.addEdge(new double[] { fDist.dist(a, b) }, a, b);
			}
		}

		pnlGrid = new GridPanel(grid, fDist);
		pnlGraph = new GraphPanel(graph, ga);

		if (ga != null && ga.length == 2) {
			pnlGraph.setGraphLayout(GraphPanel.Layout.Geo);
		}

		// maybe it would be nicer to use a MapPanel and to draw the network on a "real" map
		class NSL implements NeuronSelectedListener<double[]> {
			Grid2D<double[]> grid;

			NSL(Grid2D<double[]> grid) {
				this.grid = grid;
			}

			@Override
			public void neuronSelectedOccured(NeuronSelectedEvent<double[]> evt) {
				GridPos gp = grid.getPositionOf(evt.getNeuron());

				if (selectedColors.containsKey(gp) && selectedColors.get(gp) == selectedColor)
					selectedColors.remove(gp);
				else
					selectedColors.put(gp, selectedColor);
				
				updatePanels();
			}
		}
		pnlGraph.addNeuronSelectedListener(new NSL(grid));

		actionPerformed(new ActionEvent(gridComboBox, 0, DISTANCE));
		updatePanels();
		
		pnlGrid.addMouseListener(pnlGrid);
		pnlGrid.addNeuronSelectedListener(this);
		mapPanel.addNeuronSelectedListener(this);

		cards = new JPanel(new CardLayout());
		cards.add(pnlGrid, GRID);
		cards.add(pnlGraph, GRAPH);

		add(gridComboBox, "split 5");
		add(colorModeBox, "");
		add(quantileButton,"");
		add(gridModeComboBox, "");
		add(btnExpGrid, "");
				
		add(colorChooser, "split 3");
		//add(selectSingle, "");
		add(clearSelect, "");
		add(btnExpMap, "pushx, wrap");
		
		add( cards, "w 50%, pushy, grow");
		add( mapPanel, "grow, wrap");
		add( infoField,"span 2, growx");
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		super.actionPerformed(e);
		if (e.getSource() == gridComboBox) { // som-visualization-change
			if( !quantileButton.isEnabled() )
				quantileButton.setEnabled(true);
			if (gridComboBox.getSelectedItem() == DISTANCE) { // dmatrix

				DistanceDialog dd = new DistanceDialog(parent, "Distance...", true, gDist != null);
				if( dd.okPressed ) {
					DistMode dm = dd.getDistMode();
					StatMode sm = dd.getStatMode();
	
					neuronValues = new HashMap<GridPos, Double>();
					for (GridPos p : grid.getPositions()) {
						double[] v = grid.getPrototypeAt(p);
	
						DescriptiveStatistics ds = new DescriptiveStatistics();
						for (GridPos np : grid.getNeighbours(p)) {
							if (dm == DistMode.Normal)
								ds.addValue(fDist.dist(v, grid.getPrototypeAt(np)));
							else
								ds.addValue(gDist.dist(v, grid.getPrototypeAt(np)));
						}
						if (sm == StatMode.Mean)
							neuronValues.put(p, ds.getMean());
						else if (sm == StatMode.Median)
							neuronValues.put(p, ds.getPercentile(0.5));
						else if (sm == StatMode.Variance)
							neuronValues.put(p, ds.getVariance());
						else if (sm == StatMode.Min)
							neuronValues.put(p, ds.getMin());
						else if (sm == StatMode.Max)
							neuronValues.put(p, ds.getMax());
					}
				} else {
					gridComboBox.setSelectedItem(currentGridComboBoxItem);
				}
			} else if (gridComboBox.getSelectedItem() == RANDOM) { // random
				int k = 0;
				List<GridPos> pos = new ArrayList<GridPos>(grid.getPositions());
				Collections.shuffle(pos);
				neuronValues = new HashMap<GridPos, Double>();
				for (GridPos p : pos)
					neuronValues.put(p, (double) k++);
			} else if (gridComboBox.getSelectedItem() == CLUSTER) { // cluster
				ClusterDialogGrid cd = new ClusterDialogGrid(parent, CLUSTER, true, true);

				if (cd.isOkPressed()) {
					parent.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));

					List<double[]> prototypes = new ArrayList<double[]>(grid.getPrototypes());
					List<Set<double[]>> clusters = null;

					// connected map
					Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
					if (cd.getConnected()) {
						for (GridPos p : grid.getPositions()) {
							double[] v = grid.getPrototypeAt(p);
							Set<double[]> s = new HashSet<double[]>();
							for (GridPos nb : grid.getNeighbours(p))
								s.add(grid.getPrototypeAt(nb));
							cm.put(v, s);
						}
					}
					
					if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.kMeans)
						clusters = new ArrayList<Set<double[]>>(Clustering.kMeans(prototypes, cd.getNumCluster(), fDist).values());
					else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.SKATER) {
						Map<double[], Set<double[]>> mst = Clustering.getMinimumSpanningTree(cm, fDist);
						clusters = Clustering.skater(mst, cd.getNumCluster() - 1, fDist, 1);
					} else if (cd.getAlgorithm() == ClusterAlgorithm.Watershed) {
						Collection<Set<GridPos>> wsc;
						if (grid instanceof Grid2DHex) {
							wsc = SomUtils.getWatershedHex(cd.getMinimum(), cd.getMaximum(), cd.getBlur(), grid, fDist, false);
						} else { // TODO not sure if the implementation for non-hex-maps is correct
							int[][] ws = SomUtils.getWatershed(cd.getMinimum(), cd.getMaximum(), cd.getBlur(), grid, fDist, false);
							wsc = SomUtils.getClusterFromWatershed(ws, grid);
						}
						clusters = new ArrayList<Set<double[]>>();

						for (Set<GridPos> c : wsc) {
							Set<double[]> l = new HashSet<double[]>();
							for (GridPos p : c)
								l.add(grid.getPrototypeAt(p));
							clusters.add(l);
						}
					} else { // hierarchical
						Clustering.HierarchicalClusteringType type = null;
						if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.ALK)
							type = Clustering.HierarchicalClusteringType.average_linkage;
						else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.CLK)
							type = Clustering.HierarchicalClusteringType.complete_linkage;
						else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.SLK)
							type = Clustering.HierarchicalClusteringType.single_linkage;
						else if (cd.getAlgorithm() == ClusterDialogGrid.ClusterAlgorithm.Ward)
							type = Clustering.HierarchicalClusteringType.ward;
						
						log.debug("Algorithm: "+type+","+cd.getConnected() );
						Map<Set<double[]>,TreeNode> tree;
						if (cd.getConnected()) 
							tree = Clustering.getHierarchicalClusterTree(cm, fDist, type);
						else
							tree = Clustering.getHierarchicalClusterTree(prototypes, fDist, type);
						clusters = Clustering.cutTree( tree, cd.getNumCluster() );
					}

					// prototypes to samples
					/*Map<double[], Set<double[]>> nBmus = new HashMap<double[], Set<double[]>>();
					for (double d[] : prototypes) {
						GridPos p = grid.getPositionOf(d);
						if (bmus.containsKey(p))
							nBmus.put(d, bmus.get(p));
						else
							nBmus.put(d, new HashSet<double[]>());
					}
					showClusterSummary(parent, ResultPanel.prototypeClusterToDataCluster(nBmus, clusters), fDist, gDist);*/
								
					for( int i = 0; i < clusters.size(); i++ ) 
						for( double[] pt : clusters.get(i) ) 
							neuronValues.put( grid.getPositionOf(pt), (double)i);
					
					quantileButton.setEnabled(false);
					quantileButton.setSelected(false);
					parent.setCursor(Cursor.getDefaultCursor());
				} else { // ok not pressed
					gridComboBox.setSelectedItem(currentGridComboBoxItem);
				}
			} else { // components
				for (GridPos p : grid.getPositions()) {
					double[] v = grid.getPrototypeAt(p);
					neuronValues.put(p, v[gridComboBox.getSelectedIndex() - 3]); // RANDOM, DISTANCE, CLUSTER
				}
			}
			updatePanels();
			currentGridComboBoxItem = gridComboBox.getSelectedItem();

		} else if (e.getSource() == gridModeComboBox) {
			CardLayout cl = (CardLayout) (cards.getLayout());
			cl.show(cards, (String) gridModeComboBox.getSelectedItem());
		} else if (e.getSource() == btnExpGrid) {
			JFileChooser fc = new JFileChooser("output");

			fc.setFileFilter(FFilter.unitFilter);
			fc.setFileFilter(FFilter.weightFilter);
			fc.setFileFilter(FFilter.pngFilter);
			fc.setFileFilter(FFilter.epsFilter);
			fc.setFileFilter(FFilter.somXMLFilter);
			fc.setFileFilter(FFilter.graphMLFilter);

			int state = fc.showSaveDialog(this);
			if (state == JFileChooser.APPROVE_OPTION) {
				File fn = fc.getSelectedFile();
				try {
					if (fc.getFileFilter() == FFilter.pngFilter) {
						if (gridModeComboBox.getSelectedItem() == GRID)
							pnlGrid.saveImage(fn, "PNG");
						else
							pnlGraph.saveImage(fn, "PNG");
					} else if (fc.getFileFilter() == FFilter.epsFilter) {
						String s = fn.getAbsolutePath(); //TODO remove this
						s = s.replaceFirst(".eps", "_legend.eps");
						//saveLegend(ColorBrewerUtil.valuesToColors(neuronValues, (ColorBrewerUtil.ColorMode) colorComboBox.getSelectedItem()), neuronValues, new File(s), "EPS");
						
						if (gridModeComboBox.getSelectedItem() == GRID)
							pnlGrid.saveImage(fn, "EPS");
						else
							pnlGraph.saveImage(fn, "EPS");
					} else if (fc.getFileFilter() == FFilter.somXMLFilter) {
						SomUtils.saveGrid(grid, new FileOutputStream(fn));
					} else if (fc.getFileFilter() == FFilter.unitFilter) {
						SomToolboxUtils.writeUnitDescriptions(grid, samples, bmus, fDist, new FileOutputStream(fn));
					} else if (fc.getFileFilter() == FFilter.weightFilter) {
						SomToolboxUtils.writeWeightVectors(grid, new FileOutputStream(fn));
					} else if (fc.getFileFilter() == FFilter.graphMLFilter) {
						Map<double[], Double> nNV = new HashMap<double[], Double>();
						for (GridPos gp : neuronValues.keySet())
							nNV.put(grid.getPrototypeAt(gp), neuronValues.get(gp));
						NGResultPanel.writeGraphToGraphML(names, graph, nNV, toDoubleArrayMap(selectedColors), fn);
					}
				} catch (FileNotFoundException e1) {
					e1.printStackTrace();
				}
			}
		} 
	}
	
	@Override 
	protected Map<GridPos, Color> updatePanels() {
		Map<GridPos,Color> colorMap = super.updatePanels();
		pnlGrid.setGridColors(colorMap, selectedColors, neuronValues);
		pnlGraph.setGridColors(toDoubleArrayMap(colorMap), toDoubleArrayMap(selectedColors), toDoubleArrayMap(neuronValues));
		return colorMap;
	}

	private <T> Map<double[], T> toDoubleArrayMap(Map<GridPos, T> cm) {
		Map<double[], T> nm = new HashMap<double[], T>();
		for (GridPos gp : cm.keySet())
			nm.put(grid.getPrototypeAt(gp), cm.get(gp));
		return nm;
	}
	
	@Override
	public void neuronSelectedOccured(NeuronSelectedEvent<GridPos> evt) {
		GridPos gp = evt.getNeuron();
		
		// remove from cluster if click one already same-colored neuron
		if (selectedColors.containsKey(gp) && selectedColors.get(gp) == selectedColor)
			selectedColors.remove(gp);
		else { // color n neurons
			Set<GridPos> curNeurons = new HashSet<GridPos>();
			curNeurons.add(gp);
			
			// color neurons
			for( GridPos t : curNeurons )
				selectedColors.put(t, selectedColor);
		}
		
		int nr = 0;
		Set<double[]> s = new HashSet<double[]>();
		for( GridPos p : grid.getPositions() )
			if( selectedColors.get(p) == selectedColor ) {
				s.add(grid.getPrototypeAt(p));
				nr++;
			}
		double ss = DataUtils.getSumOfSquares(s, fDist);
		infoField.setText(nr+" neurons, "+ss+" sum of squares");
		updatePanels();
	}

	@Override
	public boolean isClusterVis() {
		return (String)gridComboBox.getSelectedItem() == CLUSTER;
	}
}

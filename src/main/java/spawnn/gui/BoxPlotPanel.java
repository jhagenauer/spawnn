package spawnn.gui;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.List;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JPanel;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.LegendItemCollection;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.renderer.category.BoxAndWhiskerRenderer;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;

import net.miginfocom.swing.MigLayout;

public class BoxPlotPanel extends JPanel {

	private static final long serialVersionUID = 1L;
	private static final int VISIBLE = 8;
	private int start = 0;

	CategoryPlot plot;
	private List<String> names = new ArrayList<String>();
	private List<double[]> samples = new ArrayList<double[]>();
	private int length = 0;
	
	final JButton prev, next;

	public BoxPlotPanel() {
		setLayout(new MigLayout(""));

		DefaultBoxAndWhiskerCategoryDataset dataset = new DefaultBoxAndWhiskerCategoryDataset();
		for (int i = 0; i < names.size(); i++) {
			String name = names.get(i);
			List<Double> list = new ArrayList<Double>();
			for (double[] d : samples)
				list.add(d[i]);
			dataset.add(list, "", name);
		}
		/* ChartPanel chartPanel = new ChartPanel( ChartFactory.createBoxAndWhiskerChart("", "", "", dataset, false) ); */

		CategoryAxis xAxis = new CategoryAxis("");
		NumberAxis yAxis = new NumberAxis("");
		BoxAndWhiskerRenderer renderer = new BoxAndWhiskerRenderer();
		  
	    renderer.setFillBox(true);
	    renderer.setSeriesPaint(0, Color.LIGHT_GRAY);
	    renderer.setSeriesPaint(1, Color.LIGHT_GRAY);
	    renderer.setSeriesOutlinePaint(0, Color.BLACK);
	    renderer.setSeriesOutlinePaint(1, Color.BLACK);
	    renderer.setUseOutlinePaintForWhiskers(true);  
	    renderer.setMedianVisible(true);
	    renderer.setMeanVisible(true);

		class MyCategoryPlot extends CategoryPlot {
			public MyCategoryPlot(DefaultBoxAndWhiskerCategoryDataset dataset, CategoryAxis xAxis, NumberAxis yAxis, BoxAndWhiskerRenderer renderer) {
				super(dataset, xAxis, yAxis, renderer);
			}

			@Override
			public LegendItemCollection getLegendItems() {
				return null;
			}
		}
		plot = new MyCategoryPlot(dataset, xAxis, yAxis, renderer);

		JFreeChart chart = new JFreeChart("", plot);
		ChartPanel chartPanel = new ChartPanel(chart);
		
		prev = new JButton("\u22b2Prev");	
		prev.setEnabled(false);
		next = new JButton("Next\u22b3");
		next.setEnabled(false);
		
		prev.addActionListener(new AbstractAction() {
			@Override
			public void actionPerformed(ActionEvent e) {
				start -= VISIBLE;
				start = Math.max(start, 0);
				
				prev.setEnabled(start!=0);
				next.setEnabled(start+VISIBLE<length);
				
				plot();
			}
		});
				
		next.addActionListener(new AbstractAction() {
			@Override
			public void actionPerformed(ActionEvent e) {
				start += VISIBLE;
				start = Math.min(start, length - VISIBLE);
				
				prev.setEnabled(start!=0);
				next.setEnabled(start+VISIBLE<length);
				
				plot();
			}
		});
		
		add(chartPanel, "span 2, wrap");
		add(prev,"");
		add(next,"align right");
	}

	public CategoryDataset createDataset(int start, int end) {
		DefaultBoxAndWhiskerCategoryDataset dataset = new DefaultBoxAndWhiskerCategoryDataset();
		for (int i = start; i < end; i++) {
			String name = names.get(i);
			List<Double> list = new ArrayList<Double>();
			for (double[] d : samples)
				list.add(d[i]);
			dataset.add(list, "", name);
		}
		return dataset;
	}

	public void setData(List<String> names, List<double[]> samples) {
		this.names = names;
		this.samples = samples;
		this.length = samples.get(0).length;
		
		// re-init buttons
		prev.setEnabled(start>0);
		next.setEnabled(start+VISIBLE<length);
	}

	public void plot() {
		plot.setDataset(createDataset(start, Math.min(start + VISIBLE,length)));
	}
}

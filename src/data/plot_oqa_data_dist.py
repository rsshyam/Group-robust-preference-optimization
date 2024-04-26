import pandas as pd 
import matplotlib.pyplot as plt

import neatplot
neatplot.set_style()

def plot_dataset_histogram(dataset_names, counts,name):
    # Plotting the histogram
    plt.bar(dataset_names, counts, color='blue')
    plt.xlabel('Datasets')
    plt.ylabel('Number of Data Points')
    plt.title('Number of Data Points in Each Dataset')
    plt.xticks(rotation=90)
    plt.savefig(f'oqa_{name}')


import pandas as pd
import matplotlib.pyplot as plt

def format_line(row):
    return f"oqa_{row['attribute']}_{row['group']}"

def plot_group_counts(df):
    # Group the DataFrame by 'attribute' and 'group' and get the counts
    group_counts = df.groupby(['attribute', 'group']).size().unstack(fill_value=0)
    
    # Print the groups for each attribute
    print("Groups for each attribute:")
    for attribute in group_counts.index:
        groups = group_counts.columns[group_counts.loc[attribute] > 0].tolist()
        print(f"{attribute}: {', '.join(groups)}")

   # Plot the counts with adjusted bar width and horizontal x-axis labels
    print(group_counts)
    fig, ax = plt.subplots(figsize=(25, 15))
    group_counts.plot(kind='bar', stacked=True, width=0.6, ax=ax)
    
    plt.title('Total Counts of Groups for Each Attribut;e')
    plt.xlabel('Attribute')
    plt.ylabel('Count')

    # Adjust legend position for better visibility
    plt.legend(title='Group', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Show the plot
    plt.tight_layout()  # Remove the rect parameter

    # Show the plot
    plt.savefig('oqa_counts')

    
    return group_counts


def plot_group_counts_alt(df):
    # Read CSV data into a DataFrame
    df = df
     
     # Create a dictionary to store counts for each attribute
    attribute_counts = {}

    # Create a dictionary to store groups for each attribute
    attribute_groups = {}

    # Iterate through rows and count unique groups for each attribute
    for _, row in df.iterrows():
        attribute = row['attribute']
        group = row['group']

        # Increment count for the attribute
        if attribute in attribute_counts:
            attribute_counts[attribute].add(group)
            attribute_groups[attribute].append(f"{group}")
        else:
            attribute_counts[attribute] = {group}
            attribute_groups[attribute] = [f"{group}"]

    # Plot the counts with a specified figure size
    plt.figure(figsize=(22, 17))
    attributes = list(attribute_counts.keys())
    attr_grp_name=[]
    for attribute in attributes:
        attr_grp_name.append(attribute+'_'+','.join(attribute_groups[attribute]))
    print(attr_grp_name)
    group_counts = [len(groups) for groups in attribute_counts.values()]

    # Assign a different color to each attribute
    colors = plt.cm.tab20(range(len(attributes)))

    # Plot the counts with different colors for each attribute
    bars = plt.bar(attributes, group_counts, color=colors)
    plt.xlabel('Attribute')
    plt.ylabel('Number of Groups')
    plt.title('Number of Groups per Attribute')

    # Add legend with only attribute names
    plt.legend(bars, attr_grp_name, title='Attributes', title_fontsize='large',fontsize='large')

    plt.xticks(rotation=0)
    neatplot.save_figure('oqa_counts')



def main():
    # Replace 'your_file.csv' with the actual path to your CSV file
    csv_file_path = 'groups.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Apply the format_line function to each row and create a new column 'formatted'
    df['formatted'] = df.apply(format_line, axis=1)

    # Display the formatted column
    formatted_list = df['formatted'].tolist()
    print(formatted_list)

    # Plot the total counts of groups for each attribute
    group_counts = plot_group_counts_alt(df)

    # Return the formatted list and group counts DataFrame
    return formatted_list, group_counts

if __name__ == '__main__':
    main()
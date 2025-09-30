import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import unicodedata
from datetime import datetime
import locale

# Configuração da página
st.set_page_config(
    page_title="Dashboard HIV/AIDS - Paraíba",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
    }
    .section-header {
        color: #1f4e79;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stPlotlyChart {
        height: auto !important;
    }
    .js-plotly-plot {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

def format_number(number):
    """Formata números usando ponto como separador decimal"""
    if pd.isna(number) or number == 0:
        return "0"
    return f"{number:,.1f}".replace(',', '.')

def format_integer(number):
    """Formata números inteiros usando ponto como separador de milhares"""
    if pd.isna(number) or number == 0:
        return "0"
    return f"{int(number):,}".replace(',', '.')

@st.cache_data
def load_data():
    """Carrega e processa todos os dados necessários"""
    
    def remove_accents(text):
        if isinstance(text, str):
            return ''.join(
                c for c in unicodedata.normalize('NFKD', text)
                if not unicodedata.combining(c)
            )
        return text
    
    # Dados por município
    df_mun = pd.read_csv('data/datasus_hiv_mun_2019-2024.csv', sep=';', 
                         skiprows=4, encoding='latin-1').sort_values(by='Total', ascending=False)
    df_mun = df_mun.iloc[:-4]
    df_mun['Município'] = df_mun['Município'].str.replace(r'^\d+\s+', '', regex=True)
    
    # Dados por sexo
    df_sex = pd.read_csv('data/datasus_hiv_sex_data_2019-2024.csv', sep=';', 
                         skiprows=4, encoding='latin-1').sort_values(by='Total', ascending=False)
    df_sex = df_sex.iloc[:-4]
    
    # Dados por faixa etária
    df_sex_age = pd.read_csv('data/datasus_hiv_sex_idade_2019-2024.csv', sep=';', 
                             skiprows=4, encoding='latin-1').sort_values(by='Total', ascending=False)
    df_sex_age = df_sex_age.iloc[:-4]
    
    # Dados de valor
    df_mun_val = pd.read_csv('data/datasus_hiv_mun_valor_2019-2024.csv', sep=';', 
                             skiprows=4, encoding='latin-1').sort_values(by='Total', ascending=False)
    df_mun_val = df_mun_val.iloc[:-4]
    df_mun_val['Município'] = df_mun_val['Município'].str.replace(r'^\d+\s+', '', regex=True)
    
    # População
    df_pop = pd.read_excel("data/ibge_pb_mun.xlsx", skiprows=2).sort_values(
        by='População no último censo - pessoas [2022]', ascending=False)
    df_pop = df_pop[["Município [-]", "População no último censo - pessoas [2022]"]].copy()
    df_pop = df_pop.rename(columns={"Município [-]": "Município"})
    df_pop['Município'] = df_pop['Município'].str.upper()
    
    # Normalizar nomes dos municípios
    for df in [df_mun, df_mun_val]:
        df['Município'] = df['Município'].apply(lambda x: remove_accents(str(x)).upper().strip())
    df_pop['Município'] = df_pop['Município'].apply(lambda x: remove_accents(str(x)).upper().strip())
    
    return df_mun, df_sex, df_sex_age, df_mun_val, df_pop

@st.cache_data
def load_geodata():
    """Carrega dados geoespaciais"""
    try:
        gdf_pb = gpd.read_file("data/PB_Municipios_2024.shp")
        def remove_accents(text):
            if isinstance(text, str):
                return ''.join(
                    c for c in unicodedata.normalize('NFKD', text)
                    if not unicodedata.combining(c)
                )
            return text
        gdf_pb['NM_MUN'] = gdf_pb['NM_MUN'].apply(lambda x: remove_accents(str(x)).upper().strip())
        return gdf_pb
    except:
        st.warning("Arquivo shapefile não encontrado. Mapa não será exibido.")
        return None

def prepare_temporal_data(df_sex):
    """Prepara dados temporais por sexo"""
    df_sex_plot = df_sex[df_sex['Sexo'].isin(['Masc', 'Fem'])].copy()
    cols = [col for col in df_sex_plot.columns if col not in ['Sexo', 'Total']]
    anos = [col.split('/')[0] for col in cols]
    
    sexos = ['Masc', 'Fem']
    dados_ano = {sexo: {} for sexo in sexos}
    
    for sexo in sexos:
        linha = df_sex_plot[df_sex_plot['Sexo'] == sexo]
        for col, ano in zip(cols, anos):
            valor = linha[col].values[0]
            try:
                v = float(valor)
            except:
                v = 0
            if ano not in dados_ano[sexo]:
                dados_ano[sexo][ano] = 0
            if not np.isnan(v) and v > 0:
                dados_ano[sexo][ano] += v
    
    anos_unicos = sorted(set(anos))
    
    df_temporal = pd.DataFrame({
        'Ano': anos_unicos,
        'Masculino': [dados_ano['Masc'].get(ano, 0) for ano in anos_unicos],
        'Feminino': [dados_ano['Fem'].get(ano, 0) for ano in anos_unicos]
    })
    
    return df_temporal

def prepare_yearly_data(df_sex):
    """Prepara dados anuais totais"""
    anos_bar = [str(ano) for ano in range(2019, 2025)]
    linha_total = df_sex[df_sex['Sexo'] == 'Total']
    
    valores_totais = []
    for ano in anos_bar:
        if ano in linha_total.columns:
            valor = linha_total[ano].values[0]
        else:
            cols_ano = [col for col in linha_total.columns if col.startswith(f"{ano}/")]
            soma_ano = 0
            for col in cols_ano:
                try:
                    v = float(linha_total[col].values[0])
                except:
                    v = 0
                soma_ano += v
            valor = soma_ano
        try:
            v = float(valor)
        except:
            v = 0
        valores_totais.append(v)
    
    # Calcular percentuais de variação
    percentuais = [None]
    for i in range(1, len(valores_totais)):
        anterior = valores_totais[i-1]
        atual = valores_totais[i]
        if anterior == 0:
            pct = None
        else:
            pct = ((atual - anterior) / anterior) * 100
        percentuais.append(pct)
    
    df_yearly = pd.DataFrame({
        'Ano': anos_bar,
        'Casos': valores_totais,
        'Variacao_Percentual': percentuais
    })
    
    return df_yearly

def calculate_general_metrics(df_sex, df_sex_age, year=None):
    """Calcula métricas gerais para um ano específico ou todo período"""
    
    if year is None:
        # Todo o período - usar dados totais
        total_cases = df_sex[df_sex['Sexo'] == 'Total']['Total'].values[0]
        male_cases = df_sex[df_sex['Sexo'] == 'Masc']['Total'].values[0]  
        female_cases = df_sex[df_sex['Sexo'] == 'Fem']['Total'].values[0]
        
        # Grupo mais afetado (homens 30-49) - todo período
        masc_30_39 = df_sex_age[(df_sex_age['Sexo'] == 'Masc')]['30 a 39 anos'].values[0]
        masc_40_49 = df_sex_age[(df_sex_age['Sexo'] == 'Masc')]['40 a 49 anos'].values[0]
        masc_30_49 = masc_30_39 + masc_40_49
        
        periodo_label = "2019-2024"
        
    else:
        # Ano específico
        linha_total = df_sex[df_sex['Sexo'] == 'Total']
        linha_masc = df_sex[df_sex['Sexo'] == 'Masc']
        linha_fem = df_sex[df_sex['Sexo'] == 'Fem']
        
        # Somar casos do ano específico
        year_str = str(year)
        cols_ano = [col for col in linha_total.columns if col.startswith(f"{year_str}/")]
        
        total_cases = sum([float(linha_total[col].values[0]) if linha_total[col].values[0] != '-' else 0 for col in cols_ano])
        male_cases = sum([float(linha_masc[col].values[0]) if linha_masc[col].values[0] != '-' else 0 for col in cols_ano])
        female_cases = sum([float(linha_fem[col].values[0]) if linha_fem[col].values[0] != '-' else 0 for col in cols_ano])
        
        # Para faixa etária por ano, usamos proporção do total (já que não temos breakdown anual por idade)
        if total_cases > 0:
            total_geral = df_sex[df_sex['Sexo'] == 'Total']['Total'].values[0]
            masc_30_39_total = df_sex_age[(df_sex_age['Sexo'] == 'Masc')]['30 a 39 anos'].values[0]
            masc_40_49_total = df_sex_age[(df_sex_age['Sexo'] == 'Masc')]['40 a 49 anos'].values[0]
            
            # Proporção estimada para o ano
            prop_30_49 = (masc_30_39_total + masc_40_49_total) / total_geral
            masc_30_49 = male_cases * prop_30_49
        else:
            masc_30_49 = 0
            
        periodo_label = str(year)
    
    return {
        'total_cases': total_cases,
        'male_cases': male_cases,
        'female_cases': female_cases,
        'masc_30_49': masc_30_49,
        'periodo_label': periodo_label
    }

def calculate_year_comparison(df_sex, df_sex_age, current_year):
    """Calcula variação percentual em relação ao ano anterior"""
    
    if current_year <= 2019:
        # Primeiro ano, sem comparação
        return None, None, None, None
    
    # Dados do ano atual
    current_metrics = calculate_general_metrics(df_sex, df_sex_age, current_year)
    
    # Dados do ano anterior
    previous_year = current_year - 1
    previous_metrics = calculate_general_metrics(df_sex, df_sex_age, previous_year)
    
    # Calcular variações percentuais
    def calc_variation(current, previous):
        if previous == 0:
            return None
        return ((current - previous) / previous) * 100
    
    total_variation = calc_variation(current_metrics['total_cases'], previous_metrics['total_cases'])
    male_variation = calc_variation(current_metrics['male_cases'], previous_metrics['male_cases'])
    female_variation = calc_variation(current_metrics['female_cases'], previous_metrics['female_cases'])
    masc_30_49_variation = calc_variation(current_metrics['masc_30_49'], previous_metrics['masc_30_49'])
    
    return total_variation, male_variation, female_variation, masc_30_49_variation

def create_map_plot(df_mun, df_pop, gdf_pb, year=2022):
    """Cria mapa choropleth"""
    if gdf_pb is None:
        st.error("Dados geoespaciais não disponíveis")
        return None
        
    # Filtrar dados por ano
    df_mun_year = df_mun[[col for col in df_mun.columns if (str(year) in str(col)) or (col == 'Município')]].copy()
    year_cols = [col for col in df_mun_year.columns if str(year) in str(col)]
    df_mun_year['Total'] = df_mun_year[year_cols].replace('-', 0).apply(pd.to_numeric, errors='coerce').sum(axis=1)
    
    # Primeiro, fazer merge com geodados para manter TODOS os municípios da PB
    gdf_map = gdf_pb.merge(
        df_mun_year[['Município', 'Total']],
        left_on='NM_MUN',
        right_on='Município',
        how='left'
    )
    
    # Depois fazer merge com dados de população
    gdf_map = gdf_map.merge(
        df_pop[['Município', 'População no último censo - pessoas [2022]']].rename(
            columns={'População no último censo - pessoas [2022]': 'População'}
        ),
        on='Município',
        how='left'
    )
    
    # Preencher valores nulos
    gdf_map['Total'] = gdf_map['Total'].fillna(0)
    gdf_map['População'] = gdf_map['População'].fillna(0)
    
    # Calcular casos por 100mil apenas para municípios com população >= 10000
    # Outros municípios aparecem no mapa mas com valor 0
    gdf_map['Casos_por_100mil'] = 0
    mask_pop = gdf_map['População'] >= 10000
    gdf_map.loc[mask_pop, 'Casos_por_100mil'] = (gdf_map.loc[mask_pop, 'Total'] / gdf_map.loc[mask_pop, 'População']) * 100000
    
    # Preparar dados formatados para hover
    def format_hover_info(row):
        if row['População'] < 10000 and row['População'] > 0:
            return f"{format_number(row['Casos_por_100mil'])} (Pop. < 10.000)"
        elif row['População'] == 0:
            return "Sem dados"
        else:
            return format_number(row['Casos_por_100mil'])
    
    gdf_map['Casos_formatado'] = gdf_map.apply(format_hover_info, axis=1)
    gdf_map['Total_formatado'] = gdf_map['Total'].apply(lambda x: format_integer(x) if pd.notna(x) else '0')
    gdf_map['Pop_formatada'] = gdf_map['População'].apply(lambda x: format_integer(x) if pd.notna(x) and x > 0 else 'Sem dados')
    
    # Criar mapa
    fig = px.choropleth_mapbox(
        gdf_map,
        geojson=gdf_map.geometry,
        locations=gdf_map.index,
        color='Casos_por_100mil',
        hover_name='NM_MUN',
        hover_data={
            'Casos_por_100mil': False,  # Remove coluna original
            'Total': False,             # Remove coluna original
            'População': False,         # Remove coluna original
            'Casos_formatado': ':.0s',  # Usa formatação customizada
            'Total_formatado': ':.0s',
            'Pop_formatada': ':.0s'
        },
        color_continuous_scale='YlOrRd',
        mapbox_style='carto-positron',
        zoom=7.5,
        center={'lat': -7.1, 'lon': -36.4},
        opacity=0.7,
        labels={
            'Casos_por_100mil': 'Casos por 100mil hab',
            'Casos_formatado': 'Casos por 100mil hab ',
            'Total_formatado': 'Total ',
            'Pop_formatada': 'População '
        }
    )
    
    # Customizar hover template
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' +
                     'Casos por 100mil hab : %{customdata[0]}<br>' +
                     'Total : %{customdata[1]}<br>' +
                     'População : %{customdata[2]}<br>' +
                     '<extra></extra>',
        customdata=np.column_stack((gdf_map['Casos_formatado'], 
                                   gdf_map['Total_formatado'], 
                                   gdf_map['Pop_formatada']))
    )
    
    # Destacar contornos dos municípios da Paraíba
    fig.update_traces(
        marker_line_color='#2c3e50',
        marker_line_width=1.2
    )
    
    fig.update_layout(
        title={
            'text': f'<b>Distribuição de casos de HIV/AIDS por 100 mil habitantes - PB ({year})</b><br><span style="font-size:12px;color:#666;">Análise baseada em municípios com população ≥ 10.000 habitantes (Censo 2022)</span>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f4e79'}
        },
        height=700,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    return fig

def create_top_municipalities_plot(df_mun, df_pop, year=2022, top_n=10):
    """Cria gráfico de top municípios"""
    # Processar dados por ano
    df_mun_year = df_mun[[col for col in df_mun.columns if (str(year) in str(col)) or (col == 'Município')]].copy()
    year_cols = [col for col in df_mun_year.columns if str(year) in str(col)]
    df_mun_year['Total'] = df_mun_year[year_cols].replace('-', 0).apply(pd.to_numeric, errors='coerce').sum(axis=1)
    
    # Merge com população
    df_mun_pop = df_mun_year.merge(
        df_pop[['Município', 'População no último censo - pessoas [2022]']].rename(
            columns={'População no último censo - pessoas [2022]': 'População'}
        ),
        on='Município',
        how='left'
    )
    
    # Calcular casos por 100 mil habitantes
    df_mun_pop = df_mun_pop[df_mun_pop['População'] >= 10000]
    df_mun_pop['Casos_por_100mil'] = (df_mun_pop['Total'] / df_mun_pop['População']) * 100000
    df_mun_pop = df_mun_pop.sort_values('Casos_por_100mil', ascending=False)
    
    top_mun = df_mun_pop.head(top_n)
    
    fig = px.bar(
        top_mun,
        x='Casos_por_100mil',
        y='Município',
        orientation='h',
        title=f'Top {top_n} Municípios - Casos por 100 mil habitantes (2022)',
        labels={'Casos_por_100mil': 'Casos por 100 mil habitantes'},
        color='Casos_por_100mil',
        color_continuous_scale='viridis'
    )
    
    # Customizar hover template
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" +
                      "%{x:.1f}<br>" +
                      "<extra></extra>"
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500,
        showlegend=False
    )
    
    return fig

def create_temporal_plot_specific_year(df_sex, selected_year):
    """Cria gráfico mensal específico para um ano selecionado"""
    if selected_year is None:
        return create_temporal_plot_all_years(df_sex)
    
    # Filtrar dados do ano específico
    year_str = str(selected_year)
    cols_ano = [col for col in df_sex.columns if col.startswith(f"{year_str}/")]
    
    if not cols_ano:
        return create_temporal_plot_all_years(df_sex)
    
    # Extrair dados mensais do ano
    meses = [col.split('/')[1] for col in cols_ano]
    
    # Dados por sexo
    linha_masc = df_sex[df_sex['Sexo'] == 'Masc']
    linha_fem = df_sex[df_sex['Sexo'] == 'Fem']
    
    valores_masc = []
    valores_fem = []
    
    for col in cols_ano:
        try:
            val_masc = float(linha_masc[col].values[0]) if linha_masc[col].values[0] != '-' else 0
            val_fem = float(linha_fem[col].values[0]) if linha_fem[col].values[0] != '-' else 0
        except:
            val_masc = val_fem = 0
        valores_masc.append(val_masc)
        valores_fem.append(val_fem)
    
    # Criar DataFrame
    df_monthly = pd.DataFrame({
        'Mês': meses,
        'Masculino': valores_masc,
        'Feminino': valores_fem
    })
    
    df_melted = df_monthly.melt(id_vars='Mês', value_vars=['Masculino', 'Feminino'], 
                               var_name='Sexo', value_name='Casos')
    
    fig = px.line(
        df_melted,
        x='Mês',
        y='Casos',
        color='Sexo',
        markers=True,
        title=f'Evolução mensal dos casos de HIV/AIDS por sexo - Paraíba ({selected_year})',
        color_discrete_map={'Masculino': '#1f77b4', 'Feminino': '#ff7f0e'}
    )
    
    # Personalizar hover template com : e espaços
    fig.update_traces(
        hovertemplate="Sexo: %{fullData.name}<br>" +
                      "Mês: %{x}<br>" +
                      "Casos: %{y}<br>" +
                      "<extra></extra>"
    )
    
    fig.update_layout(
        xaxis_title='Mês',
        yaxis_title='Número de casos',
        height=500
    )
    
    return fig

def create_temporal_plot_all_years(df_sex):
    """Cria gráfico anual com todos os anos (comportamento original)"""
    df_sex_plot = df_sex[df_sex['Sexo'].isin(['Masc', 'Fem'])].copy()
    cols = [col for col in df_sex_plot.columns if col not in ['Sexo', 'Total']]
    anos = [col.split('/')[0] for col in cols]
    
    sexos = ['Masc', 'Fem']
    dados_ano = {sexo: {} for sexo in sexos}
    
    for sexo in sexos:
        linha = df_sex_plot[df_sex_plot['Sexo'] == sexo]
        for col, ano in zip(cols, anos):
            valor = linha[col].values[0]
            try:
                v = float(valor)
            except:
                v = 0
            if ano not in dados_ano[sexo]:
                dados_ano[sexo][ano] = 0
            if not pd.isna(v) and v > 0:
                dados_ano[sexo][ano] += v
    
    anos_unicos = sorted(set(anos))
    
    df_temporal = pd.DataFrame({
        'Ano': anos_unicos,
        'Masculino': [dados_ano['Masc'].get(ano, 0) for ano in anos_unicos],
        'Feminino': [dados_ano['Fem'].get(ano, 0) for ano in anos_unicos]
    })
    
    df_melted = df_temporal.melt(id_vars='Ano', value_vars=['Masculino', 'Feminino'], 
                                var_name='Sexo', value_name='Casos')
    
    fig = px.line(
        df_melted,
        x='Ano',
        y='Casos',
        color='Sexo',
        markers=True,
        title='Evolução dos casos de HIV/AIDS por sexo - Paraíba (2019-2024)',
        color_discrete_map={'Masculino': '#1f77b4', 'Feminino': '#ff7f0e'}
    )
    
    # Personalizar hover template com : e espaços  
    fig.update_traces(
        hovertemplate="Sexo: %{fullData.name}<br>" +
                      "Ano: %{x}<br>" +
                      "Casos: %{y}<br>" +
                      "<extra></extra>"
    )
    
    fig.update_layout(
        xaxis_title='Ano',
        yaxis_title='Número de casos',
        height=500
    )
    
    return fig

def create_yearly_plot(df_yearly, selected_year=None):
    """Cria gráfico anual com percentuais e destaque opcional para ano selecionado"""
    fig = go.Figure()
    
    # Definir cores das barras (destacar ano selecionado)
    colors = []
    for ano in df_yearly['Ano']:
        if selected_year is not None and str(ano) == str(selected_year):
            colors.append('#ff7f0e')  # Laranja para destacar
        else:
            colors.append('#636efa')  # Azul padrão
    
    # Barras (sem texto duplicado)
    fig.add_trace(go.Bar(
        x=df_yearly['Ano'],
        y=df_yearly['Casos'],
        name='Casos',
        marker_color=colors,
        hovertemplate="Ano: %{x}<br>" +
                      "Casos: %{y}<br>" +
                      "<extra></extra>"
    ))
    
    # Linha de média (mais sutil, seguindo referência)
    media = df_yearly['Casos'].mean()
    fig.add_hline(y=media, line_dash="dash", line_color="#666666", 
                  annotation_text=f"Média ({media:.0f})",
                  annotation_position="top right")
    
    # Adicionar APENAS os percentuais de variação (limpo, seguindo padrão da segunda imagem)
    for i, row in df_yearly.iterrows():
        # Mostrar percentual APENAS para anos com variação válida (não 2019, não NaN)
        if (row['Variacao_Percentual'] is not None and 
            not pd.isna(row['Variacao_Percentual']) and 
            str(row['Ano']) != '2019'):
            
            pct_text = f"{row['Variacao_Percentual']:+.1f}%".replace('.', ',')
            color = '#28a745' if row['Variacao_Percentual'] > 0 else '#dc3545'
            
            fig.add_annotation(
                x=row['Ano'],
                y=row['Casos'] + max(df_yearly['Casos']) * 0.05,  # Posição um pouco mais alta
                text=pct_text,
                showarrow=False,
                font=dict(size=11, color=color, weight='bold')
                # Removidas bordas e fundo para visual limpo
            )
    
    fig.update_layout(
        title='Progressão anual dos casos totais de HIV/AIDS - Paraíba (2019-2024)',
        xaxis_title='Ano',
        yaxis_title='Número de casos',
        height=500,
        # Layout minimalista seguindo imagem de referência
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        # Grid bem sutil ou inexistente como na referência
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='lightgray'
        ),
        yaxis=dict(
            range=[0, max(df_yearly['Casos']) * 1.2],
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            showline=True,
            linecolor='lightgray'
        )
    )
    
    return fig

# Função removida - gráficos de distribuição etária agora são estáticos
# Mantendo apenas o gráfico temporal dinâmico que funciona corretamente

def create_age_distribution_plot_all_years(df_sex_age):
    """Cria gráfico de distribuição etária para todo período (comportamento original)"""
    df_total = df_sex_age[df_sex_age['Sexo'] == 'Total'].copy()
    df_by_sex = df_sex_age[df_sex_age['Sexo'] != 'Total'].copy()
    
    faixas_etarias = [col for col in df_sex_age.columns if col not in ['Sexo', 'Total']]
    
    # Subplot com dois gráficos
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribuição total por faixa etária', 'Distribuição por sexo e faixa etária')
    )
    
    # Gráfico total
    fig.add_trace(
        go.Bar(
            x=faixas_etarias,
            y=df_total.iloc[0][faixas_etarias].values,
            name='Total',
            marker_color='#888888',
            hovertemplate="Faixa etária: %{x}<br>" +
                          "Casos: %{y}<br>" +
                          "<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Gráfico por sexo
    colors = {'Masc': '#1f77b4', 'Fem': '#ff7f0e'}
    for i, row in df_by_sex.iterrows():
        fig.add_trace(
            go.Bar(
                x=faixas_etarias,
                y=row[faixas_etarias].values,
                name=row['Sexo'],
                marker_color=colors.get(row['Sexo'], '#888888'),
                hovertemplate="Sexo: %{fullData.name}<br>" +
                              "Faixa etária: %{x}<br>" +
                              "Casos: %{y}<br>" +
                              "<extra></extra>"
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text='Distribuição de casos por faixa etária - Paraíba (2019-2024)',
        height=500,
        showlegend=True
    )
    
    # Rotacionar labels do eixo x
    fig.update_xaxes(tickangle=45)
    
    return fig

# Interface principal
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 HIV/AIDS - Paraíba (2019-2024)</h1>', unsafe_allow_html=True)
    
    # Carregar dados
    with st.spinner('Carregando dados...'):
        df_mun, df_sex, df_sex_age, df_mun_val, df_pop = load_data()
        gdf_pb = load_geodata()
        df_temporal = prepare_temporal_data(df_sex)
        df_yearly = prepare_yearly_data(df_sex)
    
    # Sidebar
    st.sidebar.title("Filtros")
    
    # Filtro para dados gerais
    periodo_geral = st.sidebar.selectbox(
        "Período para indicadores gerais:",
        options=["Todo o período (2019-2024)", "2019", "2020", "2021", "2022", "2023", "2024"],
        index=0
    )
    
    # Extrair ano se não for todo período
    if periodo_geral == "Todo o período (2019-2024)":
        year_geral = None
    else:
        year_geral = int(periodo_geral)
    
    # Ano fixo para mapa (dados demográficos são de 2022)
    year_selected = 2022
    
    top_n = st.sidebar.slider(
        "Número de municípios no ranking:",
        min_value=5,
        max_value=20,
        value=10
    )
    
    # Informação sobre o ano fixo
    st.sidebar.info(
        "📊 **Mapa e Ranking fixos em 2022**\n\n"
        "Os dados demográficos são do Censo 2022. "
        "Para manter precisão metodológica, a análise "
        "por 100 mil habitantes usa apenas dados de 2022."
    )
    
    # Métricas gerais
    if year_geral is None:
        st.markdown('<h2 class="section-header">📊 Indicadores Gerais</h2>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h2 class="section-header">📊 Indicadores Gerais ({year_geral})</h2>', unsafe_allow_html=True)
    
    # Calcular métricas para o período selecionado
    metrics = calculate_general_metrics(df_sex, df_sex_age, year_geral)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_cases = metrics['total_cases']
    male_cases = metrics['male_cases']
    female_cases = metrics['female_cases']
    masc_30_49 = metrics['masc_30_49']
    
    # Calcular percentuais internos (composição)
    if total_cases > 0:
        male_percentage = (male_cases / total_cases) * 100
        female_percentage = (female_cases / total_cases) * 100
        percent_masc_30_49 = (masc_30_49 / total_cases) * 100
    else:
        male_percentage = 0
        female_percentage = 0
        percent_masc_30_49 = 0
    
    # Calcular variações temporais (se aplicável)
    if year_geral is not None:
        # Para ano específico, calcular variação em relação ao anterior
        total_var, male_var, female_var, masc_30_49_var = calculate_year_comparison(df_sex, df_sex_age, year_geral)
        
        # Formatar deltas
        def format_delta(variation):
            if variation is None:
                return None
            return f"{variation:+.1f}%".replace('.', ',')
        
        total_delta = format_delta(total_var)
        male_delta = format_delta(male_var)
        female_delta = format_delta(female_var)
        masc_30_49_delta = format_delta(masc_30_49_var)
        
    else:
        # Todo período, sem deltas (não há comparação temporal válida)
        total_delta = None
        male_delta = None
        female_delta = None
        masc_30_49_delta = None
    
    with col1:
        st.metric("Total de Casos", format_integer(total_cases), total_delta)
    
    with col2:
        st.metric("Casos Masculinos", format_integer(male_cases), male_delta)
    
    with col3:
        st.metric("Casos Femininos", format_integer(female_cases), female_delta)
    
    with col4:
        st.metric("Homens 30-49 anos", format_integer(masc_30_49), masc_30_49_delta)
    
    # Mapa choropleth
    st.markdown('<h2 class="section-header">🗺️ Distribuição Geográfica</h2>', unsafe_allow_html=True)
    
    if gdf_pb is not None:
        map_fig = create_map_plot(df_mun, df_pop, gdf_pb, year_selected)
        if map_fig:
            # Usar um container mais amplo para o mapa
            st.plotly_chart(map_fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
            })
    else:
        st.warning("Mapa não disponível - arquivo shapefile não encontrado")
    
    # Top municípios
    st.markdown('<h2 class="section-header">🏆 Ranking de Municípios</h2>', unsafe_allow_html=True)
    top_mun_fig = create_top_municipalities_plot(df_mun, df_pop, year_selected, top_n)
    st.plotly_chart(top_mun_fig, use_container_width=True)
    
    # Análise temporal
    st.markdown('<h2 class="section-header">📈 Análise Temporal</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        temporal_fig = create_temporal_plot_specific_year(df_sex, year_geral)
        st.plotly_chart(temporal_fig, use_container_width=True)
    
    with col2:
        yearly_fig = create_yearly_plot(df_yearly, year_geral)
        st.plotly_chart(yearly_fig, use_container_width=True)
    
    # Distribuição por faixa etária
    st.markdown('<h2 class="section-header">👥 Distribuição por Faixa Etária</h2>', unsafe_allow_html=True)
    age_fig = create_age_distribution_plot_all_years(df_sex_age)
    st.plotly_chart(age_fig, use_container_width=True)
    
    # Informações adicionais
    st.markdown('<h2 class="section-header">ℹ️ Glossário</h2>', unsafe_allow_html=True)
    
    with st.expander("Definições importantes"):
        st.markdown("""
        - **HIV (Vírus da Imunodeficiência Humana)**: Vírus que ataca o sistema imunológico, enfraquecendo as defesas do organismo e tornando-o mais vulnerável a infecções e doenças
        
        - **AIDS (Síndrome da Imunodeficiência Adquirida)**: Estágio avançado da infecção pelo HIV, caracterizado pelo comprometimento grave do sistema imunológico e surgimento de doenças oportunistas
        
        - **DataSUS**: Departamento de Informática do Sistema Único de Saúde, responsável pela coleta, processamento e disseminação de dados de saúde no Brasil
        
        - **MTCT (Mother-To-Child Transmission)**: Transmissão vertical, casos em que a doença é transmitida de mãe para filho durante a gestação, parto ou amamentação
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Fonte:** DataSUS - Sistema de Informações Hospitalares do SUS (SIH/SUS)")
    st.markdown(f"**Última atualização:** {datetime.now().strftime('%d/%m/%Y')}")

if __name__ == "__main__":
    main()

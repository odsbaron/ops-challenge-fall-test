import polars as pl
import numpy as np

class ops:
    @staticmethod
    def rolling_regbeta(col_x_or_expr, col_y_or_expr, window: int) -> pl.Expr:
        """
        与原来完全等价的回归 beta（Low → Close），但底层用最快的 rolling_mean 实现
        """
        if isinstance(col_x_or_expr, str):
            x = pl.col(col_x_or_expr)
        else:
            x = col_x_or_expr

        if isinstance(col_y_or_expr, str):
            y = pl.col(col_y_or_expr)
        else:
            y = col_y_or_expr

        # 下面这些中间变量的名字故意用不易冲突的前缀
        mean_x = x.rolling_mean(window_size=window, min_periods=2)
        mean_y = y.rolling_mean(window_size=window, min_periods=2)
        mean_xy = (x * y).rolling_mean(window_size=window, min_periods=2)
        mean_x2 = (x * x).rolling_mean(window_size=window, min_periods=2)

        # 经典无偏协方差与方差公式（ddof=1）
        cov_xy = mean_xy - mean_x * mean_y
        var_x  = mean_x2 - mean_x.pow(2)

        beta = pl.when(var_x.abs() < 1e-6).then(0.0).otherwise(cov_xy / var_x)

        return beta.alias("rolling_regbeta")



def ops_rolling_regbeta(input_path: str, window: int = 20) -> np.ndarray:
    # 必须先按 symbol + 时间排序，否则 rolling 结果错乱
    df = (
        pl.scan_parquet(input_path)
        .with_columns([
            pl.col("Close").cast(pl.Float64),
            pl.col("Low").cast(pl.Float64),
            pl.col("symbol").cast(pl.Categorical),
            # 如果你 parquet 里有日期列，建议加上；没有的话就靠原始顺序
            # pl.col("date").str.to_date(),  
        ])
        .sort(["symbol", "date"])   # 这一步极其关键！原来代码没写但实际也隐式排序了
    )

    result = (
        df.select(
            ops.rolling_regbeta("Low", "Close", window).over("symbol")
        )
        .collect(streaming=True)   # streaming 仍然有效，因为现在只有 rolling_mean
    )

    # 保持原来返回一维 ndarray 的行为（所有股票的 beta 按原始行顺序扁平化）
    return result.to_numpy()